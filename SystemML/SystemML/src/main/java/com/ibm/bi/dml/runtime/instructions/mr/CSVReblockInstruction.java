/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.Arrays;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVReblockInstruction extends ReblockInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public String  delim     = DataExpression.DEFAULT_DELIM_DELIMITER;
	public boolean hasHeader = DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW;
	public boolean fill      = DataExpression.DEFAULT_DELIM_FILL;
	public double  fillValue = DataExpression.DEFAULT_DELIM_FILL_VALUE;

	public CSVReblockInstruction(Operator op, byte in, byte out, int br,
			int bc, boolean hasHeader, String delim, boolean fll, double mv, String istr) {
		super(op, in, out, br, bc, false, istr);
		this.delim=delim;
		this.fill=fll;
		this.fillValue=mv;
		this.hasHeader=hasHeader;
	}

	public Instruction clone(byte in) 
	{
		// modify the input operand in the CSVReblock instruction
		String[] parts = this.instString.split(Instruction.OPERAND_DELIM);
		String[] in1f = parts[2].split(Instruction.DATATYPE_PREFIX);
		in1f[0] = Byte.toString(in);
		
		parts[2] = in1f[0] + Instruction.DATATYPE_PREFIX + in1f[1] + Instruction.VALUETYPE_PREFIX + in1f[2];

		StringBuilder sb = new StringBuilder();
		sb.append(parts[0]);
		for(int i=1; i<parts.length; i++) {
			sb.append(Instruction.OPERAND_DELIM);
			sb.append(parts[i]);
		}
		
		return parseInstruction(sb.toString());
		
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
		
		boolean hasHeader=Boolean.parseBoolean(s[6]);
		String delim = s[7];
		boolean fill=Boolean.parseBoolean(s[8]);
		double missingValue=Double.parseDouble(s[9]);
		return new CSVReblockInstruction(op, input, output, brlen, bclen, hasHeader, delim, fill, missingValue, str);
	}
}
