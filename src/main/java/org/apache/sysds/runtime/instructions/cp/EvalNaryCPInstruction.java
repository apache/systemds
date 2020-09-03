/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.dml.DmlSyntacticValidator;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.DataConverter;

/**
 * Eval built-in function instruction
 * Note: it supports only single matrix[double] output
 */
public class EvalNaryCPInstruction extends BuiltinNaryCPInstruction {

	public EvalNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//1. get the namespace and func
		String funcName = ec.getScalarInput(inputs[0]).getStringValue();
		if( funcName.contains(Program.KEY_DELIM) )
			throw new DMLRuntimeException("Eval calls to '"+funcName+"', i.e., a function outside "
				+ "the default "+ "namespace, are not supported yet. Please call the function directly.");
		
		// bound the inputs to avoiding being deleted after the function call
		CPOperand[] boundInputs = Arrays.copyOfRange(inputs, 1, inputs.length);
		List<String> boundOutputNames = new ArrayList<>();
		boundOutputNames.add(output.getName());

		//2. copy the created output matrix
		MatrixObject outputMO = new MatrixObject(ec.getMatrixObject(output.getName()));

		//3. lazy loading of dml-bodied builtin functions (incl. rename 
		// of function name to dml-bodied builtin scheme (data-type-specific)
		DataType dt1 = boundInputs[0].getDataType().isList() ? 
			DataType.MATRIX : boundInputs[0].getDataType();
		String funcName2 = Builtins.getInternalFName(funcName, dt1);
		if( !ec.getProgram().containsFunctionProgramBlock(null, funcName)) {
			if( !ec.getProgram().containsFunctionProgramBlock(null,funcName2) )
				compileFunctionProgramBlock(funcName, dt1, ec.getProgram());
			funcName = funcName2;
		}
		
		//obtain function block (but unoptimized version of existing functions for correctness)
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(null, funcName, false);
		
		//4. expand list arguments if needed
		CPOperand[] boundInputs2 = null;
		if( boundInputs.length == 1 && boundInputs[0].getDataType().isList()
			&& !(fpb.getInputParams().size() == 1 && fpb.getInputParams().get(0).getDataType().isList()))
		{
			ListObject lo = ec.getListObject(boundInputs[0]);
			checkValidArguments(lo.getData(), lo.getNames(), fpb.getInputParamNames());
			if( lo.isNamedList() )
				lo = reorderNamedListForFunctionCall(lo, fpb.getInputParamNames());
			boundInputs2 = new CPOperand[lo.getLength()];
			for( int i=0; i<lo.getLength(); i++ ) {
				Data in = lo.getData(i);
				String varName = Dag.getNextUniqueVarname(in.getDataType());
				ec.getVariables().put(varName, in);
				boundInputs2[i] = new CPOperand(varName, in);
			}
			boundInputs = boundInputs2;
		}
		
		//5. call the function
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(null, funcName,
			false, boundInputs, fpb.getInputParamNames(), boundOutputNames, "eval func");
		fcpi.processInstruction(ec);

		//6. convert the result to matrix
		Data newOutput = ec.getVariable(output);
		if (!(newOutput instanceof MatrixObject)) {
			MatrixBlock mb = null;
			if (newOutput instanceof ScalarObject) {
				//convert scalar to matrix
				mb = new MatrixBlock(((ScalarObject) newOutput).getDoubleValue());
			} else if (newOutput instanceof FrameObject) {
				//convert frame to matrix
				mb = DataConverter.convertToMatrixBlock(((FrameObject) newOutput).acquireRead());
				ec.cleanupCacheableData((FrameObject) newOutput);
			}
			outputMO.acquireModify(mb);
			outputMO.release();
			ec.setVariable(output.getName(), outputMO);
		}
		
		//7. cleanup of variable expanded from list
		if( boundInputs2 != null ) {
			for( CPOperand op : boundInputs2 )
				VariableCPInstruction.processRmvarInstruction(ec, op.getName());
		}
	}
	
	private static void compileFunctionProgramBlock(String name, DataType dt, Program prog) {
		//load builtin file and parse function statement block
		Map<String,FunctionStatementBlock> fsbs = DmlSyntacticValidator
			.loadAndParseBuiltinFunction(name, DMLProgram.DEFAULT_NAMESPACE);
		if( fsbs.isEmpty() )
			throw new DMLRuntimeException("Failed to compile function '"+name+"'.");
		
		// prepare common data structures, including a consolidated dml program
		// to facilitate function validation which tries to inline lazily loaded
		// and existing functions.
		DMLProgram dmlp = (prog.getDMLProg() != null) ? prog.getDMLProg() :
			fsbs.get(Builtins.getInternalFName(name, dt)).getDMLProg();
		for( Entry<String,FunctionStatementBlock> fsb : fsbs.entrySet() ) {
			if( !dmlp.getDefaultFunctionDictionary().containsFunction(fsb.getKey()) ) {
				dmlp.addFunctionStatementBlock(fsb.getKey(), fsb.getValue());
			}
			fsb.getValue().setDMLProg(dmlp);
		}
		DMLTranslator dmlt = new DMLTranslator(dmlp);
		ProgramRewriter rewriter = new ProgramRewriter(true, false);
		ProgramRewriter rewriter2 = new ProgramRewriter(false, true);
		
		// validate functions, in two passes for cross references
		for( FunctionStatementBlock fsb : fsbs.values() ) {
			dmlt.liveVariableAnalysisFunction(dmlp, fsb);
			dmlt.validateFunction(dmlp, fsb);
		}
		
		// compile hop dags, rewrite hop dags and compile lop dags
		// incl change of function calls to unoptimized functions calls
		for( FunctionStatementBlock fsb : fsbs.values() ) {
			dmlt.constructHops(fsb);
			rewriter.rewriteHopDAGsFunction(fsb, false); //rewrite and merge
			DMLTranslator.resetHopsDAGVisitStatus(fsb);
			rewriter.rewriteHopDAGsFunction(fsb, true); //rewrite and split
			DMLTranslator.resetHopsDAGVisitStatus(fsb);
			rewriter2.rewriteHopDAGsFunction(fsb, true);
			DMLTranslator.resetHopsDAGVisitStatus(fsb);
			HopRewriteUtils.setUnoptimizedFunctionCalls(fsb);
			DMLTranslator.resetHopsDAGVisitStatus(fsb);
			DMLTranslator.refreshMemEstimates(fsb);
			dmlt.constructLops(fsb);
		}
		
		// compile runtime program
		for( Entry<String,FunctionStatementBlock> fsb : fsbs.entrySet() ) {
			if( !prog.containsFunctionProgramBlock(null, fsb.getKey(), false) ) {
				FunctionProgramBlock fpb = (FunctionProgramBlock) dmlt
					.createRuntimeProgramBlock(prog, fsb.getValue(), ConfigurationManager.getDMLConfig());
				prog.addFunctionProgramBlock(null, fsb.getKey(), fpb, true); // optimized
				prog.addFunctionProgramBlock(null, fsb.getKey(), fpb, false);    // unoptimized -> eval
			}
		}
	}
	
	private static void checkValidArguments(List<Data> loData, List<String> loNames, List<String> fArgNames) {
		//check number of parameters
		int listSize = (loNames != null) ? loNames.size() : loData.size();
		if( listSize != fArgNames.size() )
			throw new DMLRuntimeException("Failed to expand list for function call "
				+ "(mismatching number of arguments: "+listSize+" vs. "+fArgNames.size()+").");
		
		//check individual parameters
		if( loNames != null ) {
			HashSet<String> probe = new HashSet<>();
			for( String var : fArgNames )
				probe.add(var);
			for( String var : loNames )
				if( !probe.contains(var) )
					throw new DMLRuntimeException("List argument named '"+var+"' not in function signature.");
		}
	}
	
	private static ListObject reorderNamedListForFunctionCall(ListObject in, List<String> fArgNames) {
		List<Data> sortedData = new ArrayList<>();
		for( String name : fArgNames )
			sortedData.add(in.getData(name));
		return new ListObject(sortedData, new ArrayList<>(fArgNames));
	}
}
