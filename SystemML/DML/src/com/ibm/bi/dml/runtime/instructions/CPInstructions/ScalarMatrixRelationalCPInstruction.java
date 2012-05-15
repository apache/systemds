package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ScalarMatrixRelationalCPInstruction extends RelationalBinaryCPInstruction{
	public ScalarMatrixRelationalCPInstruction(Operator op, 
			   								   CPOperand in1, 
			   								   CPOperand in2, 
			   								   CPOperand out, 
			   								   String istr){
		super(op, in1, in2, out, istr);
	}

	@Override
	public MatrixObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		MatrixObject mat = 
			(input1.get_dataType() == DataType.MATRIX) ? 
					pb.getMatrixVariable(input1.get_name()) 
					: pb.getMatrixVariable(input2.get_name());
					
		ScalarObject constant = 
			(input1.get_dataType() == DataType.SCALAR) ?
					pb.getScalarVariable(input1.get_name(), input1.get_valueType())
					: pb.getScalarVariable(input2.get_name(), input2.get_valueType());

		ScalarOperator sc_op = (ScalarOperator) optr;
		sc_op.setConstant(constant.getDoubleValue());
		
		String output_name = output.get_name();
		
		MatrixObject sores = mat.scalarOperations(sc_op, (MatrixObject)pb.getMatrixVariable(output_name) );
		
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores;
	}
}
