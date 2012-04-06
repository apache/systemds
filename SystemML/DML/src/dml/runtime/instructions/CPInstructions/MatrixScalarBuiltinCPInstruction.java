package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.DataType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ScalarOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MatrixScalarBuiltinCPInstruction extends BuiltinBinaryCPInstruction{
	public MatrixScalarBuiltinCPInstruction(Operator op,
											CPOperand in1,
											CPOperand in2,
											CPOperand out,
											String instr){
		super(op, in1, in2, out, 2, instr);
	}
	
	@Override 
	public Data processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		MatrixObject mat = 
			(input1.get_dataType() == DataType.MATRIX) ? 
					pb.getMatrixVariable(input1.get_name()) 
					: pb.getMatrixVariable(input2.get_name());
					
		ScalarObject constant = 
			(input1.get_dataType() == DataType.SCALAR) ?
					pb.getScalarVariable(input1.get_name(), input1.get_valueType())
					: pb.getScalarVariable(input2.get_name(), input2.get_valueType());
		
		ScalarOperator sc_op = (ScalarOperator)	optr;
		sc_op.setConstant(constant.getDoubleValue());
		
		String output_name = output.get_name();
		
		MatrixObject sores = mat.scalarOperations(sc_op, (MatrixObject)pb.getVariable(output_name) );
				
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores;
	}
}
