package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MatrixMatrixRelationalSPInstruction extends RelationalBinarySPInstruction {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixRelationalSPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out,
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
        MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
        MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		
		String output_name = output.getName();
		BinaryOperator bop = (BinaryOperator) _optr;
		
		MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
		
		ec.setMatrixOutput(output_name, resultBlock);
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
	}
}
