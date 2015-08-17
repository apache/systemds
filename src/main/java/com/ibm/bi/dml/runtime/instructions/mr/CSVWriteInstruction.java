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

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVWriteInstruction extends UnaryMRInstructionBase{

	
	public String delim= DataExpression.DEFAULT_DELIM_DELIMITER;
	public String header=null;//if null or empty string, then no header
	public boolean sparse=DataExpression.DEFAULT_DELIM_SPARSE;
	
	public CSVWriteInstruction(Operator op, byte in, byte out, String del, String hdr, boolean sps, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.CSVWrite;
		delim=del;
		header=hdr;
		sparse=sps;
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
		
		String header=s[4];
		String delim=s[5];
		boolean sparse=Boolean.parseBoolean(s[6]);
		return new CSVWriteInstruction(op, input, output, delim, header, sparse, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("CSVWriteInstruction.processInstruction should never be called");		
	}
}
