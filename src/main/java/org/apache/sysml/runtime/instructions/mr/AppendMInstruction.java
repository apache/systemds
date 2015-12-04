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
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class AppendMInstruction extends AppendInstruction implements IDistributedCacheConsumer
{	
	private long _offset = -1; 
	
	public AppendMInstruction(Operator op, byte in1, byte in2, long offset, CacheType type, byte out, boolean cbind, String istr)
	{
		super(op, in1, in2, out, cbind, istr);
		_offset = offset;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionParts ( str );
		InstructionUtils.checkNumFields(parts, 6);
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long offset = (long)(Double.parseDouble(parts[3]));
		byte out = Byte.parseByte(parts[4]);
		CacheType type = CacheType.valueOf(parts[5]);
		boolean cbind = Boolean.parseBoolean(parts[6]);
		
		return new AppendMInstruction(null, in1, in2, offset, type, out, cbind, str);
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
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input1);
		if( blkList == null ) 
			return;
		
		//right now this only deals with appending matrix with number of column <= blockColFactor
		for(IndexedMatrixValue in1 : blkList)
		{
			if(in1 == null)
				continue;
		
			//check for boundary block
			int blen = _cbind ? blockColFactor : blockRowFactor;
			long lastBlockColIndex = (long)Math.ceil((double)_offset/blen);	
			
			//case 1: pass through of non-boundary blocks
			MatrixIndexes ix = in1.getIndexes();
			if( (_cbind?ix.getColumnIndex():ix.getRowIndex())!=lastBlockColIndex ) {
				cachedValues.add(output, in1);
			}
			//case 2: pass through full input block and rhs block 
			else if( _cbind && in1.getValue().getNumColumns() == blen 
					|| !_cbind && in1.getValue().getNumRows() == blen ) {
				//output lhs block
				cachedValues.add(output, in1);
				
				//output shallow copy of rhs block
				DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
				if( _cbind ) {
					cachedValues.add(output, new IndexedMatrixValue(
							new MatrixIndexes(ix.getRowIndex(), ix.getColumnIndex()+1),
							dcInput.getDataBlock((int)ix.getRowIndex(), 1).getValue()));
				}
				else {
					cachedValues.add(output, new IndexedMatrixValue(
							new MatrixIndexes(ix.getRowIndex()+1, ix.getColumnIndex()),
							dcInput.getDataBlock(1, (int)ix.getColumnIndex()).getValue()));	
				}
			}
			//case 3: append operation on boundary block
			else 
			{
				DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
				
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(2);
				IndexedMatrixValue first=cachedValues.holdPlace(output, valueClass);
				first.getIndexes().setIndexes(ix);
				outlist.add(first);
				
				MatrixValue value_in2 = null;
				if( _cbind ) {
					value_in2 = dcInput.getDataBlock((int)ix.getRowIndex(), 1).getValue();
					if(in1.getValue().getNumColumns()+value_in2.getNumColumns()>blen) {
						IndexedMatrixValue second=cachedValues.holdPlace(output, valueClass);
						second.getIndexes().setIndexes(ix.getRowIndex(), ix.getColumnIndex()+1);
						outlist.add(second);
					}
				}
				else { //rbind
					value_in2 = dcInput.getDataBlock(1, (int)ix.getRowIndex()).getValue();
					if(in1.getValue().getNumRows()+value_in2.getNumRows()>blen) {
						IndexedMatrixValue second=cachedValues.holdPlace(output, valueClass);
						second.getIndexes().setIndexes(ix.getRowIndex()+1, ix.getColumnIndex());
						outlist.add(second);
					}
				}
	
				OperationsOnMatrixValues.performAppend(in1.getValue(), value_in2, outlist, 
					blockRowFactor, blockColFactor, _cbind, true, 0);			
			}
		}
	}
}
