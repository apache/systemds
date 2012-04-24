package dml.runtime.instructions.CPInstructions;

import dml.runtime.matrix.operators.Operator;

public class ComputationCPInstruction extends CPInstruction {

	public CPOperand output;
	public CPOperand input1, input2, input3;
	
	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand out ) {
		super(op);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out ) {
		super(op);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.get_name();
	}
}
