/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MatrixReshapeCPInstruction extends UnaryCPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CPOperand _opRows = null;
	private CPOperand _opCols = null;
	private CPOperand _opByRow = null;
	
	public MatrixReshapeCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String istr)
	{
		super(op, in1, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MatrixReshape;
		
		_opRows = in2;
		_opCols = in3;
		_opByRow = in4;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand out = new CPOperand(parts[5]);
			 
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeInstruction: " + str);
		else
			return new MatrixReshapeCPInstruction(new Operator(true), in1, in2, in3, in4, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock in = ec.getMatrixInput(input1.getName());
		int rows = (int)ec.getScalarInput(_opRows.getName(), _opRows.getValueType(), _opRows.isLiteral()).getLongValue(); //save cast
		int cols = (int)ec.getScalarInput(_opCols.getName(), _opCols.getValueType(), _opCols.isLiteral()).getLongValue(); //save cast
		BooleanObject byRow = (BooleanObject) ec.getScalarInput(_opByRow.getName(), ValueType.BOOLEAN, _opByRow.isLiteral());

		//execute operations 
		MatrixBlock out = new MatrixBlock();
		out = LibMatrixReorg.reshape(in, out, rows, cols, byRow.getBooleanValue());
		
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), out);
		ec.releaseMatrixInput(input1.getName());
	}
	
}
