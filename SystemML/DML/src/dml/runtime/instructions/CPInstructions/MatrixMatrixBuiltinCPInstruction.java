package dml.runtime.instructions.CPInstructions;

import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MatrixMatrixBuiltinCPInstruction extends BuiltinBinaryCPInstruction{
	public MatrixMatrixBuiltinCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, 2, istr);
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		MatrixObject mat1 = pb.getMatrixVariable(input1.get_name());
		MatrixObject mat2 = pb.getMatrixVariable(input2.get_name());
		
		BinaryOperator bop = (BinaryOperator) optr;
		
		String output_name = output.get_name();
		MatrixObject sores = mat1.binaryOperations(bop, mat2, (MatrixObject)pb.getVariable(output_name));
		
		//pb.setVariable(output_name, sores);
		pb.setVariableAndWriteToHDFS(output_name, sores);

		return sores;		
	}
}