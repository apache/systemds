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
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class AppendGInstruction extends AppendInstruction 
{
	private long _offset = -1; //cols of input1 
	private long _offset2 = -1; //cols of input2
	private long _len = -1;
	
	public AppendGInstruction(Operator op, byte in1, byte in2, long offset, long offset2, byte out, boolean cbind, String istr)
	{
		super(op, in1, in2, out, cbind, istr);
		_offset = offset;
		_offset2 = offset2;
		_len = _offset + _offset2;
	}

	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionParts ( str );
		InstructionUtils.checkNumFields (parts, 6);
			
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long offset = (long)(Double.parseDouble(parts[3]));
		long len = (long)(Double.parseDouble(parts[4]));
		byte out = Byte.parseByte(parts[5]);
		boolean cbind = Boolean.parseBoolean(parts[6]);
			
		return new AppendGInstruction(null, in1, in2, offset, len, out, cbind, str);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//setup basic meta data
		int blen = _cbind ? bclen : brlen;
		
		//Step 1: handle first input (forward blocks, change dim of last block)
		ArrayList<IndexedMatrixValue> blkList1 = cachedValues.get(input1);
		if( blkList1 != null )
			for( IndexedMatrixValue in1 : blkList1 )
			{
				if( in1 == null )
					continue;

				if( _offset%blen == 0 ) { //special case: forward only
					cachedValues.add(output, in1);	
				}
				else //general case: change dims and forward
				{	
					MatrixIndexes tmpix = in1.getIndexes();
					MatrixBlock tmpval = (MatrixBlock) in1.getValue(); //always block
					if( _cbind && _offset/blen+1 == tmpix.getColumnIndex() //border block
						|| !_cbind && _offset/blen+1 == tmpix.getRowIndex()) 
					{
						IndexedMatrixValue data = cachedValues.holdPlace(output, valueClass);
						MatrixBlock tmpvalNew = (MatrixBlock)data.getValue(); //always block
						int lrlen = _cbind ? tmpval.getNumRows() : Math.min(blen, (int)(_len-(tmpix.getRowIndex()-1)*blen));
				        int lclen = _cbind ? Math.min(blen, (int)(_len-(tmpix.getColumnIndex()-1)*blen)) : tmpval.getNumColumns();
						tmpvalNew.reset(lrlen, lclen);
						tmpvalNew.copy(0, tmpval.getNumRows()-1, 0, tmpval.getNumColumns()-1, tmpval, true);
						data.getIndexes().setIndexes(tmpix);
					}
					else //inner block
					{
						cachedValues.add(output, in1); 
					}	
				}
			}
		
		//Step 2: handle second input (split/forward blocks with new index)
		ArrayList<IndexedMatrixValue> blkList2 = cachedValues.get(input2);
		if( blkList2 != null ) 
			for( IndexedMatrixValue in2 : blkList2 )
			{
				if( in2 == null )
					continue;

				MatrixIndexes tmpix = in2.getIndexes();
				MatrixBlock tmpval = (MatrixBlock) in2.getValue(); //always block
				
				if( _offset%bclen == 0 ) //special case no split
				{
					IndexedMatrixValue data = cachedValues.holdPlace(output, valueClass);
					MatrixIndexes ix1 = data.getIndexes();
					long rix = _cbind ? tmpix.getRowIndex() : _offset/blen + tmpix.getRowIndex();
					long cix = _cbind ? _offset/blen + tmpix.getColumnIndex() : tmpix.getColumnIndex();
					ix1.setIndexes(rix, cix);
					data.set(ix1, in2.getValue());
				}
				else //general case: split and forward
				{	
					IndexedMatrixValue data1 = cachedValues.holdPlace(output, valueClass);
					MatrixIndexes ix1 = data1.getIndexes();
					MatrixBlock tmpvalNew = (MatrixBlock)data1.getValue(); //always block
					
					if( _cbind )
					{
						//first half
						int cix1 = (int)(_offset/blen + tmpix.getColumnIndex());
						int cols1 = Math.min(blen, (int)(_len-(long)(cix1-1)*blen));
						ix1.setIndexes( tmpix.getRowIndex(), cix1);
						tmpvalNew.reset( tmpval.getNumRows(), cols1 );
						tmpvalNew.copy(0, tmpval.getNumRows()-1, (int)((_offset+1)%blen)-1, cols1-1, 
								       tmpval.sliceOperations(0, tmpval.getNumRows()-1, 0, 
								    		                     (int)(cols1-((_offset)%blen)-1), new MatrixBlock()), true);
						data1.getIndexes().setIndexes(ix1);
						
						if( cols1-((_offset)%blen)<tmpval.getNumColumns() ) 
						{
							//second half (if required)
							IndexedMatrixValue data2 = cachedValues.holdPlace(output, valueClass);
							MatrixIndexes ix2 = data2.getIndexes();
							MatrixBlock tmpvalNew2 = (MatrixBlock)data2.getValue(); //always block
							int cix2 = (int)(_offset/blen + 1 + tmpix.getColumnIndex());
							int cols2 = Math.min(blen, (int)(_len-(long)(cix2-1)*blen));
							ix2.setIndexes( tmpix.getRowIndex(), cix2);
							tmpvalNew2.reset( tmpval.getNumRows(), cols2 );
							tmpvalNew2.copy(0, tmpval.getNumRows()-1, 0, cols2-1, 
									       tmpval.sliceOperations(0, tmpval.getNumRows()-1, (int)(cols1-((_offset)%blen)), 
									    		                     tmpval.getNumColumns()-1, new MatrixBlock()), true);
							data2.getIndexes().setIndexes(ix2);
						}	
					}
					else //rbind
					{
						//first half
						int rix1 = (int)(_offset/blen + tmpix.getRowIndex());
						int rows1 = Math.min(blen, (int)(_len-(long)(rix1-1)*blen));
						ix1.setIndexes( rix1, tmpix.getColumnIndex());
						tmpvalNew.reset( rows1, tmpval.getNumColumns() );
						tmpvalNew.copy((int)((_offset+1)%blen)-1, rows1-1, 0, tmpval.getNumColumns()-1,  
								       tmpval.sliceOperations(0,(int)(rows1-((_offset)%blen)-1), 
								    		   0, tmpval.getNumColumns()-1, new MatrixBlock()), true);
						data1.getIndexes().setIndexes(ix1);
						
						if( rows1-((_offset)%blen)<tmpval.getNumRows() ) 
						{
							//second half (if required)
							IndexedMatrixValue data2 = cachedValues.holdPlace(output, valueClass);
							MatrixIndexes ix2 = data2.getIndexes();
							MatrixBlock tmpvalNew2 = (MatrixBlock)data2.getValue(); //always block
							int rix2 = (int)(_offset/blen + 1 + tmpix.getRowIndex());
							int rows2 = Math.min(blen, (int)(_len-(long)(rix2-1)*blen));
							ix2.setIndexes(rix2, tmpix.getColumnIndex());
							tmpvalNew2.reset( rows2, tmpval.getNumColumns() );
							tmpvalNew2.copy(0, rows2-1, 0, tmpval.getNumColumns()-1,  
									       tmpval.sliceOperations((int)(rows1-((_offset)%blen)), tmpval.getNumRows()-1, 
									    		   0, tmpval.getNumColumns()-1, new MatrixBlock()), true);
							data2.getIndexes().setIndexes(ix2);
						}	
					}
				}
			}
	}
}
