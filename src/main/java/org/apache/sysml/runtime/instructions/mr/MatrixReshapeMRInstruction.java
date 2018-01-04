/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;

public class MatrixReshapeMRInstruction extends UnaryInstruction {
	private boolean _byrow = false;
	private MatrixCharacteristics _mcIn = null;
	private MatrixCharacteristics _mcOut = null;

	private ArrayList<IndexedMatrixValue> _cache = null;

	private MatrixReshapeMRInstruction(Operator op, byte in, long rows, long cols, boolean byrow, byte out,
			String istr) {
		super(MRType.MMTSJ, op, in, out, istr);
		instString = istr;
		_mcOut = new MatrixCharacteristics(rows, cols, -1, -1);
		_byrow = byrow;
	}

	public void setMatrixCharacteristics( MatrixCharacteristics mcIn, MatrixCharacteristics mcOut )
	{
		_mcIn = mcIn;
	}

	public static MatrixReshapeMRInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		long rows = UtilFunctions.toLong(Double.parseDouble(parts[2])); //save cast
		long cols = UtilFunctions.toLong(Double.parseDouble(parts[3])); //save cast
		boolean byrow = Boolean.parseBoolean(parts[4]);
		byte out = Byte.parseByte(parts[5]);
		 
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeMRInstruction: " + str);
		else
			return new MatrixReshapeMRInstruction(new Operator(true), in, rows, cols, byrow, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int brlen, int bclen )
		throws DMLRuntimeException 
	{		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for(IndexedMatrixValue imv : blkList)
			{
				if( imv == null )
					continue;
				
				//get cached blocks
				ArrayList<IndexedMatrixValue> out = _cache;
	
				//process instruction
				_mcOut.setBlockSize(brlen, bclen);
				out = LibMatrixReorg.reshape(imv, _mcIn, out, _mcOut, _byrow);
				
				//put the output values in the output cache
				for( IndexedMatrixValue outBlk : out )
					cachedValues.add(output, outBlk);
				
				//put blocks into own cache
				if( LibMatrixReorg.ALLOW_BLOCK_REUSE )
					_cache = out;	
			}
	}
	
	public long getNumRows() {
		return _mcOut.getRows();
	}
	
	public long getNumColunms() {
		return _mcOut.getCols();
	}
}
