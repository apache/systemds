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

package org.tugraz.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Arrays;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.rewrite.ProgramRewriter;
import org.tugraz.sysds.parser.DMLProgram;
import org.tugraz.sysds.parser.DMLTranslator;
import org.tugraz.sysds.parser.FunctionStatementBlock;
import org.tugraz.sysds.parser.dml.DmlSyntacticValidator;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.util.DataConverter;

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
		ArrayList<String> boundOutputNames = new ArrayList<>();
		boundOutputNames.add(output.getName());
		ArrayList<String> boundInputNames = new ArrayList<>();
		for (CPOperand input : boundInputs) {
			boundInputNames.add(input.getName());
		}

		//2. copy the created output matrix
		MatrixObject outputMO = new MatrixObject(ec.getMatrixObject(output.getName()));

		//3. lazy loading of dml-bodied builtin functions
		if( !ec.getProgram().containsFunctionProgramBlock(null, funcName) ) {
			FunctionProgramBlock fpb = compileFunctionProgramBlock(
				funcName, boundInputs[0].getDataType(), ec.getProgram());
			ec.getProgram().addFunctionProgramBlock(null, funcName, fpb);
		}
		
		//4. call the function
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(null, funcName);
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(null, funcName,
			boundInputs, boundInputNames, fpb.getInputParamNames(), boundOutputNames, "eval func");
		fcpi.processInstruction(ec);

		//5. convert the result to matrix
		Data newOutput = ec.getVariable(output);
		if (newOutput instanceof MatrixObject) {
			return;
		}
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
	
	private static FunctionProgramBlock compileFunctionProgramBlock(String name, DataType dt, Program prog) {
		//load builtin file and parse function statement block
		FunctionStatementBlock fsb = DmlSyntacticValidator
			.loadAndParseBuiltinFunction(name, DMLProgram.DEFAULT_NAMESPACE, dt);
		
		// validate function (could be avoided for performance because known builtin functions)
		DMLProgram dmlp = fsb.getDMLProg();
		DMLTranslator dmlt = new DMLTranslator(dmlp);
		dmlt.liveVariableAnalysisFunction(dmlp, fsb);
		dmlt.validateFunction(dmlp, fsb);
		
		// compile hop dags, rewrite hop dags and compile lop dags
		dmlt.constructHops(fsb);
		ProgramRewriter rewriter = new ProgramRewriter(true, false);
		rewriter.rewriteHopDAGsFunction(fsb, false); //rewrite and merge
		DMLTranslator.resetHopsDAGVisitStatus(fsb);
		rewriter.rewriteHopDAGsFunction(fsb, true); //rewrite and split
		DMLTranslator.resetHopsDAGVisitStatus(fsb);
		ProgramRewriter rewriter2 = new ProgramRewriter(false, true);
		rewriter2.rewriteHopDAGsFunction(fsb, true);
		DMLTranslator.resetHopsDAGVisitStatus(fsb);
		DMLTranslator.refreshMemEstimates(fsb);
		dmlt.constructLops(fsb);
		
		// compile runtime program
		return (FunctionProgramBlock) dmlt.createRuntimeProgramBlock(
			prog, fsb, ConfigurationManager.getDMLConfig());
	}
}
