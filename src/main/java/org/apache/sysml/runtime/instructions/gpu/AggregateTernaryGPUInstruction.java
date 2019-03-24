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

package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;

public class AggregateTernaryGPUInstruction extends GPUInstruction {

	private CPOperand _input1 = null;
	private CPOperand _input2 = null;
	private CPOperand _input3 = null;
	private CPOperand _output = null;
	
	private AggregateTernaryGPUInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String istr) {
		super(op, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.AggregateTernary;
		_input1 = in1;
		_input2 = in1;
		_input3 = in1;
		_output = out;
	}

	public static AggregateTernaryGPUInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("tak+*") || opcode.equalsIgnoreCase("tack+*") ) {
			InstructionUtils.checkNumFields( parts, 4 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			
			AggregateTernaryOperator op = InstructionUtils.parseAggregateTernaryOperator(opcode, 1);
			return new AggregateTernaryGPUInstruction(op, in1, in2, in3, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateTernaryGPUInstruction.parseInstruction():: Unknown opcode " + opcode);
		}		
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		GPUContext gCtx = ec.getGPUContext(0);
		String instName = getExtendedOpcode();
		AggregateTernaryOperator ab_op = (AggregateTernaryOperator) _optr;
		MatrixObject in1 = getMatrixInputForGPUInstruction(ec, _input1.getName());
		MatrixObject in2 = getMatrixInputForGPUInstruction(ec, _input2.getName());
		
		BinaryOperator bop = new BinaryOperator(Multiply.getMultiplyFnObject());
		
		int rlenA = LibMatrixCUDA.toInt(in1.getNumRows());
		int rlenB = LibMatrixCUDA.toInt(in2.getNumRows());
		int clenA = LibMatrixCUDA.toInt(in1.getNumColumns());
		int clenB = LibMatrixCUDA.toInt(in2.getNumColumns());
		int rlenOut = Math.max(rlenA, rlenB);
		int clenOut = Math.max(clenA, clenB);
		int sizeOfOutput =  rlenOut*clenOut;
		Pointer out = gCtx.allocate(instName, sizeOfOutput*LibMatrixCUDA.sizeOfDataType);
	
		// out = in1 * in2
		Pointer A = LibMatrixCUDA.getDensePointer(gCtx, in1, instName); 
		Pointer B = LibMatrixCUDA.getDensePointer(gCtx, in2, instName);
		LibMatrixCUDA.denseMatrixMatrixOp(gCtx, instName, A, B, rlenA, clenA, rlenB, clenB, out, bop);
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		
		if(!_input3.isLiteral()) {
			// out = out * in3
			MatrixObject in3 = getMatrixInputForGPUInstruction(ec, _input3.getName());
			rlenB = LibMatrixCUDA.toInt(in3.getNumRows());
			clenB = LibMatrixCUDA.toInt(in3.getNumColumns());
			if(rlenB*clenB > sizeOfOutput) {
				throw new DMLRuntimeException("Matrix-vector AggregateTernaryGPUInstruction is not supported.");
			}
			B = LibMatrixCUDA.getDensePointer(gCtx, in3, instName);
			LibMatrixCUDA.denseMatrixMatrixOp(gCtx, instName, out, B, rlenA, clenA, rlenB, clenB, out, bop);
			ec.releaseMatrixInputForGPUInstruction(_input3.getName());
		}
		
		if( _output.getDataType().isScalar() ) {
			// sum( in1*in2*in3 )
			double result = LibMatrixCUDA.reduceAll(gCtx, instName, "reduce_sum", out, sizeOfOutput);
			ec.setScalarOutput(_output.getName(), new DoubleObject(result));
		}
		else {
			// colSum( in1*in2*in3 )
			Pointer out1 = LibMatrixCUDA.getDensePointer(gCtx, 
					LibMatrixCUDA.getDenseMatrixOutputForGPUInstruction(ec, instName, _output.getName(), 1, clenOut), instName);
			LibMatrixCUDA.reduceCol(gCtx, instName, "reduce_col_sum", out, out1, rlenOut, clenOut);
			ec.releaseMatrixOutputForGPUInstruction(_output.getName());
		}
		
		gCtx.cudaFreeHelper(instName, out, gCtx.EAGER_CUDA_FREE);
	}
}
