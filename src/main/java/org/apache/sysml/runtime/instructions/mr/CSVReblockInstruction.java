/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.mr;

import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class CSVReblockInstruction extends ReblockInstruction 
{
	
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
