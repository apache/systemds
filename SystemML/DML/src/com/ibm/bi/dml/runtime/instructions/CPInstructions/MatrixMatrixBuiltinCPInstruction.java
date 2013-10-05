/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MatrixMatrixBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixBuiltinCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, 2, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
        MatrixBlock matBlock1 = ec.getMatrixInput(input1.get_name());
        MatrixBlock matBlock2 = ec.getMatrixInput(input2.get_name());
		
		String output_name = output.get_name();
		BinaryOperator bop = (BinaryOperator) optr;
		
		MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
		
		ec.setMatrixOutput(output_name, resultBlock);
		
		resultBlock = matBlock1 = matBlock2 = null;
		ec.releaseMatrixInput(input1.get_name());
		ec.releaseMatrixInput(input2.get_name());
	}
}