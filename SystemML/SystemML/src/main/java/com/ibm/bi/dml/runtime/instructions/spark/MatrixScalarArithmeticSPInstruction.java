/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class MatrixScalarArithmeticSPInstruction extends ArithmeticBinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixScalarArithmeticSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//sanity check opcode
		String opcode = getOpcode();
		if ( !(opcode.equalsIgnoreCase("+") || opcode.equalsIgnoreCase("-") || opcode.equalsIgnoreCase("*")
			 || opcode.equalsIgnoreCase("/") || opcode.equalsIgnoreCase("%%") || opcode.equalsIgnoreCase("%/%")
			 || opcode.equalsIgnoreCase("^") || opcode.equalsIgnoreCase("^2")
			 || opcode.equalsIgnoreCase("*2")) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in instruction: " + opcode);
		}

		//common binary matrix-scalar process instruction
		super.processMatrixScalarBinaryInstruction(ec);
	}
}
