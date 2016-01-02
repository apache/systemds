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

import org.apache.sysml.lops.AppendM.CacheType;
import org.apache.sysml.lops.BinaryM.VectorType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class BinaryMInstruction extends BinaryMRInstructionBase implements IDistributedCacheConsumer
{	
	private VectorType _vectorType = null;
	
	public BinaryMInstruction(Operator op, byte in1, byte in2, CacheType ctype, VectorType vtype, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
		instString = istr;
		
		_vectorType = vtype;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static BinaryMInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		CacheType ctype = CacheType.valueOf(parts[4]);
		VectorType vtype = VectorType.valueOf(parts[5]);
		
		BinaryOperator bop = InstructionUtils.parseExtendedBinaryOperator(opcode);
		return new BinaryMInstruction(bop, in1, in2, ctype, vtype, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input1);
		if( blkList == null ) 
			return;
		
		for(IndexedMatrixValue in1 : blkList)
		{
			//allocate space for the output value
			//try to avoid coping as much as possible
			IndexedMatrixValue out;
			if( (output!=input1 && output!=input2) )
				out=cachedValues.holdPlace(output, valueClass);
			else
				out=tempValue;
			
			//get second 
			DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
			IndexedMatrixValue in2 = null;
			if( _vectorType == VectorType.COL_VECTOR )
				in2 = dcInput.getDataBlock((int)in1.getIndexes().getRowIndex(), 1);
			else //_vectorType == VectorType.ROW_VECTOR
				in2 = dcInput.getDataBlock(1, (int)in1.getIndexes().getColumnIndex());
			
			//process instruction
			out.getIndexes().setIndexes(in1.getIndexes());
			OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1.getValue(), 
					in2.getValue(), out.getValue(), ((BinaryOperator)optr));
			
			//put the output value in the cache
			if(out==tempValue)
				cachedValues.add(output, out);
		}
	}

	@Override //IDistributedCacheConsumer
	public boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		return (index==input2 && index!=input1);
	}
	
	@Override //IDistributedCacheConsumer
	public void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		indexes.add(input2);
	}
}
