package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


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
		
		String[] in1f = s[2].split(Instruction.DATATYPE_PREFIX);
		input=Byte.parseByte(in1f[0]);
		
		String[] outf = s[3].split(Instruction.DATATYPE_PREFIX);
		output=Byte.parseByte(outf[0]);
		
		int brlen=Integer.parseInt(s[4]);
		int bclen=Integer.parseInt(s[5]);
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
