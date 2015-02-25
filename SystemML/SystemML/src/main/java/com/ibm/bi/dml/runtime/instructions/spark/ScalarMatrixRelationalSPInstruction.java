package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

public class ScalarMatrixRelationalSPInstruction extends RelationalBinarySPInstruction  {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarMatrixRelationalSPInstruction(Operator op, 
			   								   CPOperand in1, 
			   								   CPOperand in2, 
			   								   CPOperand out, 
			   								   String opcode,
			   								   String istr){
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		CPOperand mat = ( input1.getDataType() == DataType.MATRIX ) ? input1 : input2;
		CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
		
		MatrixBlock matBlock = ec.getMatrixInput(mat.getName());
		ScalarObject constant = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(), scalar.isLiteral());
		
		ScalarOperator sc_op = (ScalarOperator) _optr;
		sc_op.setConstant(constant.getDoubleValue());
		
		MatrixBlock resultBlock = (MatrixBlock) matBlock.scalarOperations(sc_op, new MatrixBlock());
		
		ec.releaseMatrixInput(mat.getName());
		ec.setMatrixOutput(output.getName(), resultBlock);
	}
}
