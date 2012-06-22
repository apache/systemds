package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MatrixBuiltinCPInstruction extends BuiltinUnaryCPInstruction{
	public MatrixBuiltinCPInstruction(Operator op,
									  CPOperand in,
									  CPOperand out,
									  String instr){
		super(op, in, out, 1, instr);
	}

	@Override 
	public void processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		MatrixBlock matBlock = pb.getMatrixInput(input1.get_name());
		UnaryOperator u_op = (UnaryOperator) optr;
		String output_name = output.get_name();
		
		MatrixBlock resultBlock = (MatrixBlock) (matBlock.unaryOperations(u_op, new MatrixBlock()));
		
		pb.setMatrixOutput(output_name, resultBlock);

		resultBlock = matBlock = null;
		pb.releaseMatrixInput(input1.get_name());
	}
}
