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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class BinaryOOCInstruction extends ComputationOOCInstruction {
	
	protected BinaryOOCInstruction(OOCType type, Operator bop, 
			CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, bop, in1, in2, out, opcode, istr);
	}

	public static BinaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		Operator bop = InstructionUtils.parseExtendedBinaryOrBuiltinOperator(opcode, in1, in2);
		
		return new BinaryOOCInstruction(
			OOCType.Binary, bop, in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		if (input1.isMatrix() && input2.isMatrix())
			processMatrixMatrixInstruction(ec);
		else
			processScalarMatrixInstruction(ec);
	}

	protected void processMatrixMatrixInstruction(ExecutionContext ec) {
		MatrixObject m1 = ec.getMatrixObject(input1);
		MatrixObject m2 = ec.getMatrixObject(input2);

		OOCStream<IndexedMatrixValue> qIn1 = m1.getStreamHandle();
		OOCStream<IndexedMatrixValue> qIn2 = m2.getStreamHandle();
		OOCStream<IndexedMatrixValue> qOut = new SubscribableTaskQueue<>();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		joinOOC(qIn1, qIn2, qOut, (tmp1, tmp2) -> {
			IndexedMatrixValue tmpOut = new IndexedMatrixValue();
			tmpOut.set(tmp1.getIndexes(),
				tmp1.getValue().binaryOperations((BinaryOperator)_optr, tmp2.getValue(), tmpOut.getValue()));
			return tmpOut;
		}, IndexedMatrixValue::getIndexes);
	}

	protected void processScalarMatrixInstruction(ExecutionContext ec) {
		//get operator and scalar
		CPOperand scalar = input1.isMatrix() ? input2 : input1;
		ScalarObject constant = ec.getScalarInput(scalar);
		ScalarOperator sc_op = ((ScalarOperator)_optr).setConstant(constant.getDoubleValue());

		//create thread and process binary operation
		MatrixObject min = ec.getMatrixObject(input1.isMatrix() ? input1 : input2);
		OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();
		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		mapOOC(qIn, qOut, tmp -> {
			IndexedMatrixValue tmpOut = new IndexedMatrixValue();
			tmpOut.set(tmp.getIndexes(),
				tmp.getValue().scalarOperations(sc_op, new MatrixBlock()));
			return tmpOut;
		});
	}
}
