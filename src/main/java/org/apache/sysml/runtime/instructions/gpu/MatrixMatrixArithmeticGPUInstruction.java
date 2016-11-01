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
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.Statistics;

public class MatrixMatrixArithmeticGPUInstruction extends ArithmeticBinaryGPUInstruction
{
	
	public MatrixMatrixArithmeticGPUInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		Statistics.incrementNoOfExecutedGPUInst();
		
		MatrixObject in1 = ec.getMatrixInputForGPUInstruction(_input1.getName());
		MatrixObject in2 = ec.getMatrixInputForGPUInstruction(_input2.getName());
		
		//TODO: make hop level changes for this
		boolean isLeftTransposed = false;
		boolean isRightTransposed = false;
		int rlen = isLeftTransposed ? (int) in1.getNumColumns() : (int) in1.getNumRows();
		int clen = isLeftTransposed ? (int) in1.getNumRows() : (int) in1.getNumColumns();
		
		ec.setMetaData(_output.getName(), rlen, clen);
		
		BinaryOperator bop = (BinaryOperator) _optr;
		LibMatrixCUDA.bincellOp(ec, in1, in2, _output.getName(), isLeftTransposed, isRightTransposed, bop);
		
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
        ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}