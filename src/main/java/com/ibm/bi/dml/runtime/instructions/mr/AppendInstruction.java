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


import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AppendInstruction extends BinaryMRInstructionBase 
{
		
	/**
	 * 
	 * @param op
	 * @param in1
	 * @param in2
	 * @param out
	 * @param istr
	 */
	public AppendInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		instString = istr;	
		mrtype = MRINSTRUCTION_TYPE.Append;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if( opcode.equals("mappend") )
			return AppendMInstruction.parseInstruction(str);
		else if( opcode.equals("rappend") )
			return AppendRInstruction.parseInstruction(str);
		else if( opcode.equals("gappend") )
			return AppendGInstruction.parseInstruction(str);
		else
			throw new DMLRuntimeException("Unsupported append operation code: "+opcode);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		throw new DMLUnsupportedOperationException("Operations on base append instruction not supported.");
	}
}
