package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MatrixMatrixRelationalCPInstruction extends RelationalBinaryCPInstruction{
	public MatrixMatrixRelationalCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, istr);
	}

	@Override
	public void processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
        MatrixBlock matBlock1 = pb.getMatrixInput(input1.get_name());
        MatrixBlock matBlock2 = pb.getMatrixInput(input2.get_name());
		
		String output_name = output.get_name();
		BinaryOperator bop = (BinaryOperator) optr;
		
		MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
		
		pb.setMatrixOutput(output_name, resultBlock);
		
		resultBlock = matBlock1 = matBlock2 = null;
		pb.releaseMatrixInput(input1.get_name());
		pb.releaseMatrixInput(input2.get_name());
	}
}