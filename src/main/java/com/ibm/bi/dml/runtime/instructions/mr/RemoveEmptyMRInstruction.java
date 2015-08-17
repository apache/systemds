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

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


/**
 * Supported optcodes: rmempty.
 * 
 */
public class RemoveEmptyMRInstruction extends BinaryInstruction
{	
	
	
	private long    _len   = -1;
	private boolean _rmRows = true;
	
	
	public RemoveEmptyMRInstruction(Operator op, byte in1, byte in2, long len, boolean rmRows, byte out, String istr)
	{
		super(op, in1, in2, out, istr);
		instString = istr;
		
		_len = len;
		_rmRows = rmRows;
	}
	
	public boolean isRemoveRows()
	{
		return _rmRows;
	}
	
	public long getOutputLen()
	{
		return _len;
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
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		
		if(!opcode.equalsIgnoreCase("rmempty"))
			throw new DMLRuntimeException("Unknown opcode while parsing an RemoveEmptyMRInstruction: " + str);
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long rlen = UtilFunctions.toLong(Double.parseDouble(parts[3]));
		boolean rmRows = parts[4].equals("rows");
		byte out = Byte.parseByte(parts[5]);
		
		return new RemoveEmptyMRInstruction(null, in1, in2, rlen, rmRows, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{			
		//get input and offsets
		IndexedMatrixValue inData = cachedValues.getFirst(input1);
		IndexedMatrixValue inOffset = cachedValues.getFirst(input2);

		//execute remove empty operations
		ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
		LibMatrixReorg.rmempty(inData, inOffset, _rmRows, _len, blockRowFactor, blockColFactor, out);
		
		//put results into cache map
		for( IndexedMatrixValue imv : out )
			cachedValues.add(output, imv);
	}
}
