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

public class MatrixBVectorBuiltinSPInstruction extends BuiltinBinarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixBVectorBuiltinSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) 
		throws DMLRuntimeException 
	{
		super(op, in1, in2, out, opcode, istr);
		
		//sanity check opcodes
		if(!( opcode.equalsIgnoreCase("mapmax") || opcode.equalsIgnoreCase("mapmin")) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in MatrixBVectorBuiltinSPInstruction: " + toString());
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//common binary matrix-broadcast vector process instruction
		super.processMatrixBVectorBinaryInstruction(ec);
	}
}
