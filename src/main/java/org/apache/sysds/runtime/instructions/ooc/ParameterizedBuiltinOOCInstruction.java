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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

import java.util.LinkedHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class ParameterizedBuiltinOOCInstruction extends ComputationOOCInstruction {

	protected final LinkedHashMap<String, String> params;

	protected ParameterizedBuiltinOOCInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
		String opcode, String istr) {
		super(OOCInstruction.OOCType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}

	public static ParameterizedBuiltinOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand(parts[parts.length - 1]);

		// process remaining parts and build a hash map
		LinkedHashMap<String, String> paramsMap = ParameterizedBuiltinCPInstruction.constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;

		if(opcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinOOCInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			return new ParameterizedBuiltinOOCInstruction(null, paramsMap, out, opcode, str);
		}
		else
			throw new NotImplementedException(); // TODO
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(instOpcode.equalsIgnoreCase(Opcodes.REPLACE.toString())) {
			if(ec.isFrameObject(params.get("target"))){
				throw new NotImplementedException();
			} else{
				MatrixObject targetObj = ec.getMatrixObject(params.get("target"));
				OOCStream<IndexedMatrixValue> qIn = targetObj.getStreamHandle();
				OOCStream<IndexedMatrixValue> qOut = createWritableStream();

				double pattern = Double.parseDouble(params.get("pattern"));
				double replacement = Double.parseDouble(params.get("replacement"));

				mapOOC(qIn, qOut, tmp -> new IndexedMatrixValue(tmp.getIndexes(), tmp.getValue().replaceOperations(new MatrixBlock(), pattern, replacement)));

				ec.getMatrixObject(output).setStreamHandle(qOut);
			}
		}
		else if(instOpcode.equalsIgnoreCase(Opcodes.CONTAINS.toString())) {
			MatrixObject targetObj = ec.getMatrixObject(params.get("target"));
			OOCStream<IndexedMatrixValue> qIn = targetObj.getStreamHandle();
			Data pattern = ec.getVariable(params.get("pattern"));

			if( pattern == null ) //literal
				pattern = ScalarObjectFactory.createScalarObject(Types.ValueType.FP64, params.get("pattern"));

			if (!pattern.getDataType().isScalar())
				throw new NotImplementedException();

			Data finalPattern = pattern;

			addInStream(qIn);
			addOutStream(); // This instruction has no output stream

			CompletableFuture<Boolean> future = new CompletableFuture<>();

			filterOOC(qIn, tmp -> {
				boolean contains = ((MatrixBlock)tmp.getValue()).containsValue(((ScalarObject)finalPattern).getDoubleValue());

				if (contains)
					future.complete(true);
			}, tmp -> !future.isDone(), // Don't start a separate worker if result already known
				() -> future.complete(false));     // Then the pattern was not found

			boolean ret;
			try {
				ret = future.get();
			} catch (InterruptedException | ExecutionException e) {
				throw new DMLRuntimeException(e);
			}

			ec.setScalarOutput(output.getName(), new BooleanObject(ret));
		}
	}
}
