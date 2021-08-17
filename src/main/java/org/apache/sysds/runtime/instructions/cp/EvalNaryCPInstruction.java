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
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.dml.DmlSyntacticValidator;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.ProgramConverter;

/**
 * Eval built-in function instruction
 * Note: it supports only single matrix[double] output
 */
public class EvalNaryCPInstruction extends BuiltinNaryCPInstruction {

	private int _threadID = -1;
	
	public EvalNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//1. get the namespace and func
		String funcName = ec.getScalarInput(inputs[0]).getStringValue();
		String nsName = null; //default namespace
		if( funcName.contains(Program.KEY_DELIM) ) {
			String[] parts = DMLProgram.splitFunctionKey(funcName);
			funcName = parts[1];
			nsName = parts[0];
		}
		
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
		if( !ec.getProgram().containsFunctionProgramBlock(nsName, funcName)) {
			//error handling non-existing functions
			if( !Builtins.contains(funcName, true, false) //builtins and their private functions
				&& !ec.getProgram().containsFunctionProgramBlock(DMLProgram.BUILTIN_NAMESPACE, funcName2)) {
				String msgNs = (nsName==null) ? DMLProgram.DEFAULT_NAMESPACE : nsName;
				throw new DMLRuntimeException("Function '" 
					+ DMLProgram.constructFunctionKey(msgNs, funcName)+"' (called through eval) is non-existing.");
			}
			//load and compile missing builtin function
			nsName = DMLProgram.BUILTIN_NAMESPACE;
			synchronized(ec.getProgram()) { //prevent concurrent recompile/prog modify
				if( !ec.getProgram().containsFunctionProgramBlock(nsName, funcName2) )
					compileFunctionProgramBlock(funcName, dt1, ec.getProgram());
			}
			funcName = funcName2;
		}
		
		//obtain function block (but unoptimized version of existing functions for correctness)
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(nsName, funcName, false);
		
		//copy function block in parfor context (avoid excessive thread contention on recompilation)
		if( ProgramBlock.isThreadID(_threadID) && ParForProgramBlock.COPY_EVAL_FUNCTIONS ) {
			String funcNameParfor = funcName + Lop.CP_CHILD_THREAD + _threadID;
			if( !ec.getProgram().containsFunctionProgramBlock(nsName, funcNameParfor, false) ) { //copy on demand
				fpb = ProgramConverter.createDeepCopyFunctionProgramBlock(fpb, new HashSet<>(), new HashSet<>());
				ec.getProgram().addFunctionProgramBlock(nsName, funcNameParfor, fpb, false);
			}
			fpb = ec.getProgram().getFunctionProgramBlock(nsName, funcNameParfor, false);
			funcName = funcNameParfor;
		}
		
		//4. expand list arguments if needed
		CPOperand[] boundInputs2 = null;
		LineageItem[] lineageInputs = null;
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
			lineageInputs = DMLScript.LINEAGE 
					? lo.getLineageItems().toArray(new LineageItem[lo.getLength()]) : null;
		}
		
		//5. call the function (to unoptimized function)
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(nsName, funcName,
			false, boundInputs, lineageInputs, fpb.getInputParamNames(), boundOutputNames, "eval func");
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
			else {
				throw new DMLRuntimeException("Invalid eval return type: "+newOutput.getDataType().name()
					+ " (valid: matrix/frame/scalar; where frames or scalars are converted to output matrices)");
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
	
	@Override
	public void updateInstructionThreadID(String pattern, String replace) {
		//obtain thread (parfor worker) ID from replacement string
		_threadID = Integer.parseInt(replace.substring(Lop.CP_CHILD_THREAD.length()));
	}
	
	private static void compileFunctionProgramBlock(String name, DataType dt, Program prog) {
		//load builtin file and parse function statement block
		String nsName = DMLProgram.BUILTIN_NAMESPACE;
		Map<String,FunctionStatementBlock> fsbs = DmlSyntacticValidator
			.loadAndParseBuiltinFunction(name, nsName);
		if( fsbs.isEmpty() )
			throw new DMLRuntimeException("Failed to compile function '"+name+"'.");
		
		DMLProgram dmlp = (prog.getDMLProg() != null) ? prog.getDMLProg() :
			fsbs.get(Builtins.getInternalFName(name, dt)).getDMLProg();
		
		//filter already existing functions (e.g., already loaded internally-called functions)
		fsbs = (dmlp.getBuiltinFunctionDictionary() == null) ? fsbs : fsbs.entrySet().stream()
			.filter(e -> !dmlp.getBuiltinFunctionDictionary().containsFunction(e.getKey()))
			.collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
		
		// prepare common data structures, including a consolidated dml program
		// to facilitate function validation which tries to inline lazily loaded
		// and existing functions.
		for( Entry<String,FunctionStatementBlock> fsb : fsbs.entrySet() ) {
			dmlp.createNamespace(nsName); // create namespace on demand
			dmlp.addFunctionStatementBlock(nsName, fsb.getKey(), fsb.getValue());
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
			FunctionProgramBlock fpb = (FunctionProgramBlock) dmlt
				.createRuntimeProgramBlock(prog, fsb.getValue(), ConfigurationManager.getDMLConfig());
			prog.addFunctionProgramBlock(nsName, fsb.getKey(), fpb, true);  // optimized
			prog.addFunctionProgramBlock(nsName, fsb.getKey(), fpb, false); // unoptimized -> eval
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
		List<LineageItem> sortedLI = DMLScript.LINEAGE ? new ArrayList<>() : null;
		for( String name : fArgNames ) {
			sortedData.add(in.getData(name));
			if (DMLScript.LINEAGE)
				sortedLI.add(in.getLineageItem(name));
		}
		return new ListObject(sortedData, new ArrayList<>(fArgNames), sortedLI);
	}
}
