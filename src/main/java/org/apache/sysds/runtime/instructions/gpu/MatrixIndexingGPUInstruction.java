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
package org.apache.sysds.runtime.instructions.gpu;

import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.GPUStatistics;

public class MatrixIndexingGPUInstruction extends GPUInstruction {
	CPOperand rowLower, rowUpper, colLower, colUpper;
	CPOperand input1;
	CPOperand input2;
	CPOperand output;

	private MatrixIndexingGPUInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(null, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.MatrixIndexing;
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
		input1 = in;
		output = out;
	}

	private MatrixIndexingGPUInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(op, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.MatrixIndexing;
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
		input1 = lhsInput;
		input2 = rhsInput;
		output = out;
	}

	public static MatrixIndexingGPUInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(RightIndex.OPCODE) ) {
			if ( parts.length == 7 ) {
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand(parts[1]);
				rl = new CPOperand(parts[2]);
				ru = new CPOperand(parts[3]);
				cl = new CPOperand(parts[4]);
				cu = new CPOperand(parts[5]);
				out = new CPOperand(parts[6]);
				if( in.getDataType()==DataType.MATRIX )
					return new MatrixIndexingGPUInstruction(in, rl, ru, cl, cu, out, opcode, str);
				else 
					throw new DMLRuntimeException("Can index only on Matrices in GPU");
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase(LeftIndex.OPCODE)) {
			if ( parts.length == 8 ) {
				CPOperand lhsInput, rhsInput, rl, ru, cl, cu, out;
				lhsInput = new CPOperand();
				rhsInput = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				lhsInput.split(parts[1]);
				rhsInput.split(parts[2]);
				rl.split(parts[3]);
				ru.split(parts[4]);
				cl.split(parts[5]);
				cu.split(parts[6]);
				out.split(parts[7]);
				if( lhsInput.getDataType()==DataType.MATRIX )
					return new MatrixIndexingGPUInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
				else 
					throw new DMLRuntimeException("Can index only on Matrices in GPU");
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingGPUInstruction: " + str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		String opcode = getOpcode();
		
		IndexRange ixrange = getIndexRange(ec);
		if ( opcode.equalsIgnoreCase(RightIndex.OPCODE) ) {
			MatrixObject mat1 = getMatrixInputForGPUInstruction(ec, input1.getName());
			LibMatrixCUDA.sliceOperations(ec, ec.getGPUContext(0), getExtendedOpcode(), mat1, ixrange, output.getName());
			ec.releaseMatrixInputForGPUInstruction(input1.getName());
			ec.releaseMatrixOutputForGPUInstruction(output.getName());
		}
		else {
			throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
		}
	}
	
	IndexRange getIndexRange(ExecutionContext ec) {
		return new IndexRange( //rl, ru, cl, ru
			(int)(ec.getScalarInput(rowLower).getLongValue()-1),
			(int)(ec.getScalarInput(rowUpper).getLongValue()-1),
			(int)(ec.getScalarInput(colLower).getLongValue()-1),
			(int)(ec.getScalarInput(colUpper).getLongValue()-1));
	}
}