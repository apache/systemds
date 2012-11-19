package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MatrixMatrixArithmeticCPInstruction extends ArithmeticBinaryCPInstruction{
	public MatrixMatrixArithmeticCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, istr);
	}
	
	@Override
	public void processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{

		long begin, st, tread, tcompute, twrite, ttotal;
		
		begin = System.currentTimeMillis();
		// Read input matrices
        MatrixBlock matBlock1 = pb.getMatrixInput(input1.get_name());
        MatrixBlock matBlock2 = pb.getMatrixInput(input2.get_name());
		tread = System.currentTimeMillis() - begin;
        
		st = System.currentTimeMillis();
		BinaryOperator bop = (BinaryOperator) optr;
		String output_name = output.get_name();
		// Perform computation using input matrices, and produce the result matrix
		MatrixBlock soresBlock = (MatrixBlock) (matBlock1.binaryOperations (bop, matBlock2, new MatrixBlock()));
		tcompute = System.currentTimeMillis() - st;
        
		st = System.currentTimeMillis();
		// Release the memory occupied by input matrices
		matBlock1 = matBlock2 = null;
		pb.releaseMatrixInput(input1.get_name());
		pb.releaseMatrixInput(input2.get_name());
		// Attach result matrix with MatrixObject associated with output_name
		pb.setMatrixOutput(output_name, soresBlock);
        soresBlock = null;
		
        twrite = System.currentTimeMillis()-st;
		ttotal = System.currentTimeMillis()-begin;
		
		LOG.trace("CPInst " + this.toString() + "\t" + tread + "\t" + tcompute + "\t" + twrite + "\t" + ttotal);
	}

}