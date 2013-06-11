package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVReblockInstruction extends ReblockInstruction {

	public char delim=' ';
	public boolean ignoreFirstLine=false;
	public double missingValue=0.0;
	public CSVReblockInstruction(Operator op, byte in, byte out, int br,
			int bc, char delim, boolean ignore1Line, double mv, String istr) {
		super(op, in, out, br, bc, istr);
		this.delim=delim;
		this.missingValue=mv;
		ignoreFirstLine=ignore1Line;
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
		char delim=(char)Byte.parseByte(s[6]);
		boolean ignore=Boolean.parseBoolean(s[7]);
		double missingValue=Double.parseDouble(s[8]);
		return new CSVReblockInstruction(op, input, output, brlen, bclen, delim, ignore, missingValue, str);
	}
}
