package dml.runtime.instructions.MRInstructions;

import dml.runtime.matrix.operators.Operator;

public abstract class UnaryMRInstructionBase extends MRInstruction {

	public byte input;
	
	public UnaryMRInstructionBase(Operator op, byte in, byte out)
	{
		super(op, out);
		input=in;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input, output};
	}

}
