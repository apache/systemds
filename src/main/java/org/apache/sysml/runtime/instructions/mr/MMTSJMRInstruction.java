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

import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;


/**
 * 
 * 
 */
public class MMTSJMRInstruction extends UnaryInstruction
{	
	
	private MMTSJType _type = null;

	public MMTSJMRInstruction(Operator op, byte in, MMTSJType type, byte out, String istr)
	{
		super(op, in, out, istr);
		mrtype = MRINSTRUCTION_TYPE.MMTSJ;
		instString = istr;
		
		_type = type;
	}
	
	/**
	 * 
	 * @return
	 */
	public MMTSJType getMMTSJType()
	{
		return _type;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MMTSJMRInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		MMTSJType titype = MMTSJType.valueOf(parts[3]);
		 
		if(!opcode.equalsIgnoreCase("tsmm"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MMTIJMRInstruction: " + str);
		else
			return new MMTSJMRInstruction(new Operator(true), in, titype, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList !=null )
			for(IndexedMatrixValue imv : blkList)
			{
				if(imv==null)
					continue;
				MatrixValue in = imv.getValue();
				
				//allocate space for the output value
				IndexedMatrixValue iout = null;
				if(output==input)
					iout=tempValue;
				else
					iout=cachedValues.holdPlace(output, valueClass);
				iout.getIndexes().setIndexes(1, 1);
				MatrixValue out = iout.getValue();
				
				//process instruction
				if( in instanceof MatrixBlock && out instanceof MatrixBlock )
					((MatrixBlock) in).transposeSelfMatrixMultOperations((MatrixBlock)out, _type );
				else
					throw new DMLRuntimeException("Types "+in.getClass()+" and "+out.getClass()+" incompatible with "+MatrixBlock.class);
				
				//put the output value in the cache
				if(iout==tempValue)
					cachedValues.add(output, iout);
			}
	}
}
