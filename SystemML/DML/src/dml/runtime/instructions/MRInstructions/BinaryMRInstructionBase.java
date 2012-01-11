package dml.runtime.instructions.MRInstructions;

import dml.runtime.matrix.operators.Operator;

public abstract class BinaryMRInstructionBase extends MRInstruction {

	public byte input1, input2;
	
	public BinaryMRInstructionBase(Operator op, byte in1, byte in2, byte out)
	{
		super(op, out);
		input1=in1;
		input2=in2;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input1, input2};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input1, input2, output};
	}

}
