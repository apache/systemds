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
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.COV;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class CovarianceCPInstruction extends BinaryCPInstruction {
	
	private CovarianceCPInstruction(Operator op, CPOperand in1,
		CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr)
	{
		super(CPType.AggregateBinary, op, in1, in2, in3, out, opcode, istr);
	}

	public static CovarianceCPInstruction parseInstruction( String str )
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if( !opcode.equalsIgnoreCase("cov") )
			throw new DMLRuntimeException("CovarianceCPInstruction.parseInstruction():: Unknown opcode " + opcode);
		
		InstructionUtils.checkNumFields(parts, 4, 5); //w/o opcode
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = (parts.length==5) ? null : new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[parts.length-2]);
		int numThreads = Integer.parseInt(parts[parts.length-1]);
		
		COVOperator cov = new COVOperator(COV.getCOMFnObject(), numThreads);
		return new CovarianceCPInstruction(cov, in1, in2, in3, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
	{
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		String output_name = output.getName(); 
		COVOperator cov_op = (COVOperator)_optr;
		CM_COV_Object covobj = null;
		
		if ( input3 == null ) {
			// Unweighted: cov.mvar0.mvar1.out
			covobj = matBlock1.covOperations(cov_op, matBlock2);
			
			ec.releaseMatrixInput(input1.getName(), input2.getName());
		}
		else {
			// Weighted: cov.mvar0.mvar1.weights.out
			MatrixBlock wtBlock = ec.getMatrixInput(input3.getName());
			
			covobj = matBlock1.covOperations(cov_op, matBlock2, wtBlock);
			
			ec.releaseMatrixInput(input1.getName(), 
				input2.getName(), input3.getName());
		}
		
		double val = covobj.getRequiredResult(_optr);
		ec.setScalarOutput(output_name, new DoubleObject(val));
	}
}
