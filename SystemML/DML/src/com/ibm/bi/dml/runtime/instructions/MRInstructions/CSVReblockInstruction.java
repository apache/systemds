/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVReblockInstruction extends ReblockInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public String delim=" ";
	public boolean ignoreFirstLine=false;
	public double missingValue=0.0;
	public CSVReblockInstruction(Operator op, byte in, byte out, int br,
			int bc, String delim, boolean ignore1Line, double mv, String istr) {
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
		String delim=s[6];
		boolean ignore=Boolean.parseBoolean(s[7]);
		double missingValue=Double.parseDouble(s[8]);
		return new CSVReblockInstruction(op, input, output, brlen, bclen, delim, ignore, missingValue, str);
	}
}
