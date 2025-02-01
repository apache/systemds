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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class AggregateTernaryCPInstruction extends ComputationCPInstruction {

	private static final Log LOG = LogFactory.getLog(AggregateTernaryCPInstruction.class.getName());

	private AggregateTernaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String istr) {
		super(CPType.AggregateTernary, op, in1, in2, in3, out, opcode, istr);
	}

	public static AggregateTernaryCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) || opcode.equalsIgnoreCase(Opcodes.TACKPM.toString())) {
			InstructionUtils.checkNumFields(parts, 5);

			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			int numThreads = Integer.parseInt(parts[5]);

			AggregateTernaryOperator op = InstructionUtils.parseAggregateTernaryOperator(opcode, numThreads);
			return new AggregateTernaryCPInstruction(op, in1, in2, in3, out, opcode, str);
		}
		throw new DMLRuntimeException("AggregateTernaryInstruction.parseInstruction():: Unknown opcode " + opcode);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		MatrixBlock matBlock3 = input3.isLiteral() ? null : // matrix or literal 1
			ec.getMatrixInput(input3.getName());

		AggregateTernaryOperator ab_op = (AggregateTernaryOperator) _optr;

		validateInput(matBlock1, matBlock2, matBlock3, ab_op);
		MatrixBlock ret = MatrixBlock
			.aggregateTernaryOperations(matBlock1, matBlock2, matBlock3, new MatrixBlock(), ab_op, true);

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		if(!input3.isLiteral())
			ec.releaseMatrixInput(input3.getName());
		if(output.getDataType().isScalar())
			ec.setScalarOutput(output.getName(), new DoubleObject(ret.get(0, 0)));
		else
			ec.setMatrixOutput(output.getName(), ret);
	}

	private static void validateInput(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, AggregateTernaryOperator op) {
		int m1r = m1.getNumRows();
		int m2r = m2.getNumRows();
		int m3r = m3 == null ? m2r : m3.getNumRows();
		int m1c = m1.getNumColumns();
		int m2c = m2.getNumColumns();
		int m3c = m3 == null ? m2c : m3.getNumColumns();

		if(m1r != m2r || m1c != m2c || m2r != m3r || m2c != m3c){
			if(LOG.isTraceEnabled()){
				LOG.trace("matBlock1:" + m1);
				LOG.trace("matBlock2:" + m2);
				LOG.trace("matBlock3:" + m3);
			}
			throw new DMLRuntimeException("Invalid dimensions for aggregate ternary (" + m1r + "x" + m1c + ", "
				+ m2r + "x" + m2c + ", " + m3r + "x" + m3c + ").");
		}
				
		if(!(op.aggOp.increOp.fn instanceof KahanPlus && op.binaryFn instanceof Multiply))
			throw new DMLRuntimeException("Unsupported operator for aggregate ternary operations.");

	}
}
