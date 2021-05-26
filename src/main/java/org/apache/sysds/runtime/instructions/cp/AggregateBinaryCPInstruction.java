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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public class AggregateBinaryCPInstruction extends BinaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(AggregateBinaryCPInstruction.class.getName());

	public boolean transposeLeft;
	public boolean transposeRight;

	private AggregateBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr) {
		super(CPType.AggregateBinary, op, in1, in2, out, opcode, istr);
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

		if(!opcode.equalsIgnoreCase("ba+*")) {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

		int numFields = InstructionUtils.checkNumFields(parts, 4, 6);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		int k = Integer.parseInt(parts[4]);
		AggregateBinaryOperator aggbin = InstructionUtils.getMatMultOperator(k);
		if ( numFields == 6 ){
			boolean isLeftTransposed = Boolean.parseBoolean(parts[5]);
			boolean isRightTransposed = Boolean.parseBoolean(parts[6]);
			return new AggregateBinaryCPInstruction(aggbin, in1, in2, out, opcode, str, isLeftTransposed,
				isRightTransposed);
		}
		else return new AggregateBinaryCPInstruction(aggbin, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// get inputs
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());

		// compute matrix multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		MatrixBlock ret;

		if(matBlock1 instanceof CompressedMatrixBlock) {
			CompressedMatrixBlock main = (CompressedMatrixBlock) matBlock1;
			ret = main.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op, transposeLeft, transposeRight);
		}
		else if(matBlock2 instanceof CompressedMatrixBlock) {
			CompressedMatrixBlock main = (CompressedMatrixBlock) matBlock2;
			ret = main.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op, transposeLeft, transposeRight);
		}
		else {
			// todo move rewrite rule here. to do 
			// t(x) %*% y -> t(t(y) %*% x)
			if(transposeLeft){
				ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), ab_op.getNumThreads());
				matBlock1 = matBlock1.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
			}
			if(transposeRight){
				ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), ab_op.getNumThreads());
				matBlock2 = matBlock2.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
			}
			ret = matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op);
		}

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), ret);
	}
}
