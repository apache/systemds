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

import com.ibm.bi.dml.lops.AppendM.CacheType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AppendMInstruction extends AppendInstruction implements IDistributedCacheConsumer
{	
	private long _offset = -1; 
	
	public AppendMInstruction(Operator op, byte in1, byte in2, long offset, CacheType type, byte out, String istr)
	{
		super(op, in1, in2, out, istr);
		_offset = offset;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long offset = (long)(Double.parseDouble(parts[3]));
		byte out = Byte.parseByte(parts[4]);
		CacheType type = CacheType.valueOf(parts[5]);
		
		return new AppendMInstruction(null, in1, in2, offset, type, out, str);
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
			long lastBlockColIndex = (long)Math.ceil((double)_offset/blockColFactor);
			
			//case 1: pass through of non-boundary blocks
			if( in1.getIndexes().getColumnIndex()!=lastBlockColIndex ) {
				cachedValues.add(output, in1);
			}
			//case 2: pass through full input block and rhs block 
			else if( in1.getValue().getNumColumns() == blockColFactor ) {
				//output lhs block
				cachedValues.add(output, in1);
				
				//output shallow copy of rhs block
				DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
				IndexedMatrixValue tmp = new IndexedMatrixValue(
						new MatrixIndexes(in1.getIndexes().getRowIndex(), in1.getIndexes().getColumnIndex()+1),
						dcInput.getDataBlock((int)in1.getIndexes().getRowIndex(), 1).getValue());
				cachedValues.add(output, tmp);
			}
			//case 3: append operation on boundary block
			else 
			{
				DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
				MatrixValue value_in2 = dcInput.getDataBlock((int)in1.getIndexes().getRowIndex(), 1).getValue();
				
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(2);
				IndexedMatrixValue first=cachedValues.holdPlace(output, valueClass);
				first.getIndexes().setIndexes(in1.getIndexes());
				outlist.add(first);
				
				if(in1.getValue().getNumColumns()+value_in2.getNumColumns()>blockColFactor)
				{
					IndexedMatrixValue second=cachedValues.holdPlace(output, valueClass);
					second.getIndexes().setIndexes(in1.getIndexes().getRowIndex(), in1.getIndexes().getColumnIndex()+1);
					outlist.add(second);
				}
	
				OperationsOnMatrixValues.performAppend(in1.getValue(), value_in2, outlist, 
					blockRowFactor, blockColFactor, true, 0);			
			}
		}
	}
}
