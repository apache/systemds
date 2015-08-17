/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class ComputationSPInstruction extends SPInstruction {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

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
