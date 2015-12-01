/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class ComputationSPInstruction extends SPInstruction {
	

	public CPOperand output;
	public CPOperand input1, input2, input3;
	
	public ComputationSPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr ) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	public ComputationSPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr ) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.getName();
	}

	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException 
	 */
	protected void updateUnaryOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		updateUnaryOutputMatrixCharacteristics(sec, input1.getName(), output.getName());
	}
	
	/**
	 * 
	 * @param sec
	 * @param nameIn
	 * @param nameOut
	 * @throws DMLRuntimeException
	 */
	protected void updateUnaryOutputMatrixCharacteristics(SparkExecutionContext sec, String nameIn, String nameOut) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(nameIn);
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(nameOut);
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + mc1.toString() + " " + mcOut.toString());
			else
				mcOut.set(mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
		}
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	protected void updateBinaryOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mcIn1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcIn2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		boolean outer = (mcIn1.getRows()>1 && mcIn1.getCols()==1 && mcIn2.getRows()==1 && mcIn2.getCols()>1);
		
		if(!mcOut.dimsKnown()) {
			if(!mcIn1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + mcIn1.toString() + " " + mcIn2.toString() + " " + mcOut.toString());
			else if(outer)
				sec.getMatrixCharacteristics(output.getName()).set(mcIn1.getRows(), mcIn2.getCols(), mcIn1.getRowsPerBlock(), mcIn2.getColsPerBlock());
			else
				sec.getMatrixCharacteristics(output.getName()).set(mcIn1.getRows(), mcIn1.getCols(), mcIn1.getRowsPerBlock(), mcIn1.getRowsPerBlock());
		}
	}
}
