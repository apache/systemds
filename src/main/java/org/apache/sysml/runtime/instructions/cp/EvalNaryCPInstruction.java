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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.DataConverter;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Eval built-in function instruction
 * Note: it supports only single matrix[double] output
 */
public class EvalNaryCPInstruction extends BuiltinNaryCPInstruction {

	public EvalNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
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

		//3. call the function
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(
			null, funcName, boundInputs, boundInputNames, boundOutputNames, "eval func");
		fcpi.processInstruction(ec);

		//4. convert the result to matrix
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
}
