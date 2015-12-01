package org.apache.sysml.runtime.instructions.spark;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class MatrixScalarRelationalSPInstruction extends RelationalBinarySPInstruction  
{
	
	public MatrixScalarRelationalSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		String opcode = getOpcode();
		if ( !(opcode.equalsIgnoreCase("==") || opcode.equalsIgnoreCase("!=") || opcode.equalsIgnoreCase("<")
			  || opcode.equalsIgnoreCase(">") || opcode.equalsIgnoreCase("<=") || opcode.equalsIgnoreCase(">=")) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in instruction: " + opcode);		
		}	

		//common binary matrix-scalar process instruction
		super.processMatrixScalarBinaryInstruction(ec);
	}
}
