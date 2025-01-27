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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.parser.ConstIdentifier;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.Expression;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.StatementBlock;
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
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.ProgramConverter;

/**
 * Eval built-in function instruction
 * Note: it supports only single matrix[double] output
 */
public class EvalNaryCPInstruction extends BuiltinNaryCPInstruction {

	// default: not in parfor context; otherwise updated via 
	// updateInstructionThreadID during parfor worker setup and/or recompilation
	private int _threadID = 0;
	
	public EvalNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// There are two main types of eval function calls, which share most of the
		// code for lazy function loading and execution:
		// a) a single-return eval fcall returns a matrix which is bound to the output
		//    (if the function returns multiple objects, the first one is used as output)
		// b) a multi-return eval fcall gets all returns of the function call and
		//    creates a named list used the names of the function signature
		
		//1. get the namespace and function names
		String funcName = ec.getScalarInput(inputs[0]).getStringValue();
		String nsName = null; //default namespace
		if( funcName.contains(Program.KEY_DELIM) ) {
			String[] parts = DMLProgram.splitFunctionKey(funcName);
			funcName = parts[1];
			nsName = parts[0];
		}
		
		// bind the inputs to avoiding being deleted after the function call
		CPOperand[] boundInputs = Arrays.copyOfRange(inputs, 1, inputs.length);
		
		//2. copy the created output matrix
		MatrixObject outputMO = !output.isMatrix() ? null :
			new MatrixObject(ec.getMatrixObject(output.getName()));
		
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
				fpb = ProgramConverter.createDeepCopyFunctionProgramBlock(fpb, new HashSet<>(), new HashSet<>(), _threadID);
				ec.getProgram().addFunctionProgramBlock(nsName, funcNameParfor, fpb, false);
				ec.addTmpParforFunction(DMLProgram.constructFunctionKey(nsName, funcNameParfor));
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
			lo = lo.isNamedList() ?
				appendNamedDefaults(lo, fpb.getStatementBlock()) :
				appendPositionalDefaults(lo, fpb.getStatementBlock());
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
			lineageInputs = !DMLScript.LINEAGE ? null : 
				lo.getLineageItems().toArray(new LineageItem[lo.getLength()]);
		}
		
		// bind the outputs
		List<String> boundOutputNames = new ArrayList<>();
		if( output.getDataType().isMatrix() )
			boundOutputNames.add(output.getName());
		else //list
			boundOutputNames.addAll(fpb.getOutputParamNames());
		
		//5. call the function (to unoptimized function)
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(nsName, funcName,
			false, boundInputs, lineageInputs, fpb.getInputParamNames(), boundOutputNames, "eval func");
		fcpi.processInstruction(ec);
		
		//6a. convert the result to matrix
		if( output.getDataType().isMatrix() ) {
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
		}
		//6a. wrap outputs in named list (evalList)
		else {
			Data[] ldata = boundOutputNames.stream()
				.map(n -> ec.getVariable(n)).toArray(Data[]::new);
			String[] lnames = boundOutputNames.toArray(new String[0]);
			ListObject listOutput = null;
			if (DMLScript.LINEAGE) {
				CPOperand[] listOperands = boundOutputNames.stream().map(n -> ec.containsVariable(n) ? new CPOperand(n,
					ec.getVariable(n)) : new CPOperand(n, ValueType.STRING, DataType.SCALAR, true)).toArray(CPOperand[]::new);
				LineageItem[] liList = LineageItemUtils.getLineage(ec, listOperands);
				listOutput = new ListObject(Arrays.asList(ldata), boundOutputNames, Arrays.asList(liList));
			}
			else
				listOutput = new ListObject(ldata, lnames);
			ec.setVariable(output.getName(), listOutput);
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
			.loadAndParseBuiltinFunction(name, nsName, true); //forced for remote parfor
		if( fsbs.isEmpty() )
			throw new DMLRuntimeException("Failed to compile function '"+name+"'.");
		
		DMLProgram dmlp = (prog.getDMLProg() != null) ? prog.getDMLProg() :
			fsbs.get(Builtins.getInternalFName(name, dt)).getDMLProg();
		
		//filter already existing functions (e.g., already loaded internally-called functions)
		//note: in remote parfor the runtime program might contain more functions than the DML program
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
			//mark as conditional (warnings instead of errors) because internally
			//called functions might not be available in dmlp but prog in remote parfor
			dmlt.validateFunction(dmlp, fsb, true);
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
			if(!prog.containsFunctionProgramBlock(nsName, fsb.getKey(), true))
				prog.addFunctionProgramBlock(nsName, fsb.getKey(), fpb, true);  // optimized
			if(!prog.containsFunctionProgramBlock(nsName, fsb.getKey(), false))
				prog.addFunctionProgramBlock(nsName, fsb.getKey(), fpb, false); // unoptimized -> eval
		}
	}
	
	private static ListObject appendNamedDefaults(ListObject params, StatementBlock sb) {
		if( !params.isNamedList() || sb == null )
			return params;
		
		//best effort replacement of scalar literal defaults
		FunctionStatement fstmt = (FunctionStatement) sb.getStatement(0);
		ListObject ret = new ListObject(params);
		for( int i=0; i<fstmt.getInputParams().size(); i++ ) {
			String param = fstmt.getInputParamNames()[i];
			if( !ret.contains(param)
				&& fstmt.getInputDefaults().get(i) != null
				&& fstmt.getInputParams().get(i).getDataType().isScalar() )
			{
				ValueType vt = fstmt.getInputParams().get(i).getValueType();
				Expression expr = fstmt.getInputDefaults().get(i);
				if( expr instanceof ConstIdentifier ) {
					ScalarObject sobj = ScalarObjectFactory.createScalarObject(vt, expr.toString());
					LineageItem litem = !DMLScript.LINEAGE ? null :
						LineageItemUtils.createScalarLineageItem(ScalarObjectFactory.createLiteralOp(sobj));
					ret.add(param, sobj, litem);
				}
			}
		}
		
		return ret;
	}
	
	private static ListObject appendPositionalDefaults(ListObject params, StatementBlock sb) {
		if( sb == null )
			return params;
		
		//best effort replacement of scalar literal defaults
		FunctionStatement fstmt = (FunctionStatement) sb.getStatement(0);
		ListObject ret = new ListObject(params);
		for( int i=ret.getLength(); i<fstmt.getInputParams().size(); i++ ) {
			String param = fstmt.getInputParamNames()[i];
			if( !(fstmt.getInputDefaults().get(i) != null
				&& fstmt.getInputParams().get(i).getDataType().isScalar()
				&& fstmt.getInputDefaults().get(i) instanceof ConstIdentifier) )
				throw new DMLRuntimeException("Unable to append positional scalar default for '"+param+"'");
			ValueType vt = fstmt.getInputParams().get(i).getValueType();
			Expression expr = fstmt.getInputDefaults().get(i);
			ScalarObject sobj = ScalarObjectFactory.createScalarObject(vt, expr.toString());
			LineageItem litem = !DMLScript.LINEAGE ? null :
				LineageItemUtils.createScalarLineageItem(ScalarObjectFactory.createLiteralOp(sobj));
			ret.add(sobj, litem);
		}
		
		return ret;
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
