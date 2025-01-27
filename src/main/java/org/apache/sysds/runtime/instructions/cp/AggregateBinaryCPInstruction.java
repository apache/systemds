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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class AggregateBinaryCPInstruction extends BinaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(AggregateBinaryCPInstruction.class.getName());

	final public boolean transposeLeft;
	final public boolean transposeRight;

	private AggregateBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr) {
		super(CPType.AggregateBinary, op, in1, in2, out, opcode, istr);
		transposeLeft = false;
		transposeRight = false;
	}

	private AggregateBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr, boolean transposeLeft, boolean transposeRight) {
		super(CPType.AggregateBinary, op, in1, in2, out, opcode, istr);
		this.transposeLeft = transposeLeft;
		this.transposeRight = transposeRight;
	}

	public static AggregateBinaryCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(Opcodes.MMULT.toString())) {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

		int numFields = InstructionUtils.checkNumFields(parts, 4, 6);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		int k = Integer.parseInt(parts[4]);
		AggregateBinaryOperator op = InstructionUtils.getMatMultOperator(k);
		if(numFields == 6) {
			boolean lt = Boolean.parseBoolean(parts[5]);
			boolean rt = Boolean.parseBoolean(parts[6]);
			return new AggregateBinaryCPInstruction(op, in1, in2, out, opcode, str, lt, rt);
		}
		return new AggregateBinaryCPInstruction(op, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		// check compressed inputs
		final boolean comp1 = matBlock1 instanceof CompressedMatrixBlock;
		final boolean comp2 = matBlock2 instanceof CompressedMatrixBlock;

		if(comp1 || comp2)
			processCompressedAggregateBinary(ec, matBlock1, matBlock2, comp1, comp2);
		else if(transposeLeft || transposeRight)
			processTransposedFusedAggregateBinary(ec, matBlock1, matBlock2);
		else
			processNormal(ec, matBlock1, matBlock2);

	}

	private void processNormal(ExecutionContext ec, MatrixBlock matBlock1, MatrixBlock matBlock2) {
		// compute matrix multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		MatrixBlock ret = matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op);

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), ret);
	}

	private void processTransposedFusedAggregateBinary(ExecutionContext ec, MatrixBlock matBlock1,
		MatrixBlock matBlock2) {

		// compute matrix multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		MatrixBlock ret;

		// TODO: Use rewrite rule here t(x) %*% y -> t(t(y) %*% x)
		if(transposeLeft) {
			matBlock1 = LibMatrixReorg.transpose(matBlock1, ab_op.getNumThreads());
			ec.releaseMatrixInput(input1.getName());
		}
		if(transposeRight) {
			matBlock2 = LibMatrixReorg.transpose(matBlock2, ab_op.getNumThreads());
			ec.releaseMatrixInput(input2.getName());
		}

		ret = matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op);

		if(!transposeLeft)
			ec.releaseMatrixInput(input1.getName());
		if(!transposeRight)
			ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), ret);
	}

	private void processCompressedAggregateBinary(ExecutionContext ec, MatrixBlock matBlock1, MatrixBlock matBlock2,
		boolean c1, boolean c2) {

		// compute matrix multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		MatrixBlock ret;

		if(c1) {
			CompressedMatrixBlock main = (CompressedMatrixBlock) matBlock1;
			ret = main.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op, transposeLeft,
				transposeRight);
		}
		else {
			CompressedMatrixBlock main = (CompressedMatrixBlock) matBlock2;
			ret = main.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op, transposeLeft,
				transposeRight);
		}

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), ret);
	}
}
