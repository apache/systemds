package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ScalarMatrixArithmeticCPInstruction extends ArithmeticBinaryCPInstruction{
	public ScalarMatrixArithmeticCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, istr);
	}
	
	@Override
	public void processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		CPOperand mat, scalar;
		if ( input1.get_dataType() == DataType.MATRIX ) {
			mat = input1;
			scalar = input2;
		}
		else {
			scalar = input1;
			mat = input2;
		}
		
		MatrixBlock matBlock = (MatrixBlock) pb.getMatrixInput(mat.get_name());
		ScalarObject constant = (ScalarObject) pb.getScalarInput(scalar.get_name(), scalar.get_valueType());

		ScalarOperator sc_op = (ScalarOperator) optr;
		sc_op.setConstant(constant.getDoubleValue());
		
		String output_name = output.get_name();
		MatrixBlock resultBlock = (MatrixBlock) matBlock.scalarOperations(sc_op, new MatrixBlock());
		
		matBlock = null;
		pb.releaseMatrixInput(mat.get_name());
		pb.setMatrixOutput(output_name, resultBlock);
		resultBlock = null;
	}
}
