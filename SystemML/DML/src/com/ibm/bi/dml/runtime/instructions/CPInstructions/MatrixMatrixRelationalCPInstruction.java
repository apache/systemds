package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
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
	public MatrixObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		MatrixObject mat1 = pb.getMatrixVariable(input1.get_name());
		MatrixObject mat2 = pb.getMatrixVariable(input2.get_name());
			
		String output_name = output.get_name();
		
		BinaryOperator operator = (BinaryOperator) optr;
		
		MatrixObject sores = mat1.binaryOperations(operator, mat2, (MatrixObject)pb.getVariable(output.get_name()));

		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores;
	}
}