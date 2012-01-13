package dml.runtime.instructions.CPInstructions;

import dml.runtime.matrix.operators.Operator;

public class ScalarCPInstruction extends CPInstruction {

	public CPOperand output;
	public CPOperand input1, input2;
	
	public ScalarCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand out ) {
		super(op);
		input1 = in1;
		input2 = in2;
		output = out;
	}

	public String getOutputVariableName() {
		return output.get_name();
	}
}
