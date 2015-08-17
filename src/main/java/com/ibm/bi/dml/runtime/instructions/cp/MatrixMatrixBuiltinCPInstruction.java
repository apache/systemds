/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.LibCommonsMath;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MatrixMatrixBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixBuiltinCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, 2, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{

		String opcode = getOpcode();
        
        if ( LibCommonsMath.isSupportedMatrixMatrixOperation(opcode) ) {
        	MatrixBlock solution = LibCommonsMath.matrixMatrixOperations((MatrixObject)ec.getVariable(input1.getName()), (MatrixObject)ec.getVariable(input2.getName()), opcode);
    		ec.setMatrixOutput(output.getName(), solution);
        	return;
        }
		
        /* Default behavior of this instruction */
		String output_name = output.getName();
		BinaryOperator bop = (BinaryOperator) _optr;
		
        MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
        MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		
        MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
		
		ec.setMatrixOutput(output_name, resultBlock);
		
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
	}
	
}