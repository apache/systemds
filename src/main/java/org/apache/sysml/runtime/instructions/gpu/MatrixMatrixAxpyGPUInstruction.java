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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

public class MatrixMatrixAxpyGPUInstruction extends ArithmeticBinaryGPUInstruction
{
	
	CPOperand constant = null;
	int multiplier = 1;
	public MatrixMatrixAxpyGPUInstruction(Operator op, 
			   CPOperand in1, 
			   CPOperand constant, 
			   int multiplier,
			   CPOperand in2, 
			   CPOperand out, 
			   String opcode,
			   String istr){
		super(op, in1, in2, out, opcode, istr);
		this.constant = constant;
		this.multiplier = multiplier;
	}
	
	public static MatrixMatrixAxpyGPUInstruction parseInstruction ( String str ) throws DMLRuntimeException {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields ( parts, 4 );
		
		String opcode = parts[0];
		int multiplier = 1;
		if(opcode.equals("-*"))
			multiplier = -1;
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand constant = new CPOperand(parts[2]);
		if(constant.getDataType() != DataType.SCALAR)
			throw new DMLRuntimeException("Expected second operand to be a scalar");
		CPOperand in2 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		
		DataType dt1 = in1.getDataType();
		DataType dt2 = in2.getDataType();
		DataType dt3 = out.getDataType();
	 
		Operator operator = (dt1 != dt2) ?
				InstructionUtils.parseScalarBinaryOperator(opcode, (dt1 == DataType.SCALAR)) : 
				InstructionUtils.parseBinaryOperator(opcode);
		
		if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX && dt3 == DataType.MATRIX) {
			return new MatrixMatrixAxpyGPUInstruction(operator, in1, constant, multiplier, in2, out, opcode, str);	
		}
		else if( dt3 == DataType.MATRIX && ((dt1 == DataType.SCALAR && dt2 == DataType.MATRIX) || (dt1 == DataType.MATRIX && dt2 == DataType.SCALAR)) ) {
			throw new DMLRuntimeException("Unsupported GPU PlusMult/MinusMult ArithmeticInstruction.");
			// return new ScalarMatrixArithmeticGPUInstruction(operator, in1, in2, out, opcode, str);
		}
		else
			throw new DMLRuntimeException("Unsupported GPU ArithmeticInstruction.");
	}

	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		
		MatrixObject in1 = getMatrixInputForGPUInstruction(ec, _input1.getName());
		MatrixObject in2 = getMatrixInputForGPUInstruction(ec, _input2.getName());
		ScalarObject scalar = ec.getScalarInput(constant.getName(), constant.getValueType(), constant.isLiteral());
		
		long rlen1 = in1.getNumRows();
		long clen1 = in1.getNumColumns();
		long rlen2 = in2.getNumRows();
		long clen2 = in2.getNumColumns();
		if(isValidMMOperation(rlen1, rlen2, clen1, clen2) || isValidMVOperation(rlen1, rlen2, clen1, clen2)) {
			ec.setMetaData(_output.getName(), (int)rlen1, (int)clen1);
		}
		else { 
			throw new DMLRuntimeException("Incorrect dimensions of inputs in GPU axpy operation. input1:" + rlen1 + " X " + clen1 +
					" and input2:" + rlen2 + " X " + clen2);
		}
		
		LibMatrixCUDA.axpy(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, in2, _output.getName(), multiplier*scalar.getDoubleValue());
		
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
	
	private boolean isValidMMOperation(long rlen1, long rlen2, long clen1, long clen2) {
		return rlen1 == rlen2 && clen1 == clen2; 
	}
	
	private boolean isValidMVOperation(long rlen1, long rlen2, long clen1, long clen2) {
		return (rlen1 == rlen2 && clen2 == 1) || (rlen2 == 1 && clen1 == clen2); 
	}
	
}