package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class ReblockInstruction extends UnaryMRInstructionBase {

	public int brlen;
	public int bclen;
	public ReblockInstruction (Operator op, byte in, byte out, int br, int bc, String istr) {
		super(op, in, out);
		brlen=br;
		bclen=bc;
		instString = istr;
	}
	
	public static Instruction parseInstruction(String str) {
		Operator op = null;
		
		byte input, output;
		String[] s=str.split(Instruction.OPERAND_DELIM);
		
		String[] in1f = s[1].split(Instruction.VALUETYPE_PREFIX);
		input=Byte.parseByte(in1f[0]);
		
		String[] outf = s[2].split(Instruction.VALUETYPE_PREFIX);
		output=Byte.parseByte(outf[0]);
		
		int brlen=Integer.parseInt(s[3]);
		int bclen=Integer.parseInt(s[4]);
		return new ReblockInstruction(op, input, output, brlen, bclen, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("ReblockInstruction.processInstruction should never be called");
		
	}
	
}
