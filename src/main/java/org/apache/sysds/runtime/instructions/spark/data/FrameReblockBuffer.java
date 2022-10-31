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


package org.apache.sysds.runtime.instructions.spark.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FrameReblockBuffer
{
	//default buffer size: 5M -> 5M * 3x8B = 120MB 
	public static final int DEFAULT_BUFFER_SIZE = 5000000;
	
	private int _bufflen = -1;
	private int _count = -1;
	
	private FrameCell[] _buff = null;
	
	private long _rlen = -1;
	private long _clen = -1;
	private int _blen = -1;
	
	private ValueType[] _schema;

	public FrameReblockBuffer( int buffersize,  long rlen, long clen, ValueType[] schema )
	{
		_bufflen = buffersize;
		_count = 0;
		
		_buff = new FrameCell[ _bufflen ];
		for(int i=0; i< _bufflen; i++)
			_buff[i] = new FrameCell();
		
		_rlen = rlen;
		_clen = clen;
		_blen = Math.max((int)(_bufflen/_clen), 1);
		_blen = (int)clen;
		
		_schema = schema;
	}
	
	public int getSize() {
		return _count;
	}
	
	public int getCapacity() {
		return _bufflen;
	}

	public void appendCell( long r, long c, Object obj ) {
		_buff[_count].setRow((int)r);
		_buff[_count].setCol((int)c);
		_buff[_count].setObjVal(obj);
		_count++;
	}

	public void flushBufferToBinaryBlocks( ArrayList<Pair<Long, FrameBlock>> outList ) 
		throws DMLRuntimeException
	{
		if( _count == 0 )
			return;
		
		//Step 1) sort reblock buffer (blockwise, no in-block sorting!)
		Arrays.sort( _buff, 0 ,_count, new FrameReblockBufferComparator() );
				
		//Step 2) output blocks 
		Long tmpIx = -1L;
		FrameBlock tmpBlock = new FrameBlock(_schema);
		
		//put values into block and output
		long cbi = -1, cbj = -1; //current block indexes
		for( int i=0; i<_count; i++ )
		{
			//compute block indexes (w/ robustness for meta data handling)
			long bi = Math.max(UtilFunctions.computeBlockIndex(_buff[i].getRow(), _blen), 1);
			long bj = UtilFunctions.computeBlockIndex(_buff[i].getCol(), _blen);
			
			//output block and switch to next index pair
			if( bi != cbi || bj != cbj ) {
				if( cbi != -1 && cbj != -1)
					outputBlock(outList, tmpIx, tmpBlock);
				cbi = bi;
				cbj = bj;
				tmpIx = (bi-1)*_blen+1;
				tmpBlock = new FrameBlock(_schema);
				tmpBlock.ensureAllocatedColumns(Math.min(_blen, (int)(_rlen-(bi-1)*_blen)));
			}
			
			int ci = UtilFunctions.computeCellInBlock(_buff[i].getRow(), _blen);
			int cj = UtilFunctions.computeCellInBlock(_buff[i].getCol(), _blen);
			if( ci == -3 )
				tmpBlock.getColumnMetadata(cj).setMvValue(_buff[i].getObjVal().toString());
			else if( ci == -2 )
				tmpBlock.getColumnMetadata(cj).setNumDistinct(Long.parseLong(_buff[i].getObjVal().toString()));
			else
				tmpBlock.set(ci, cj, _buff[i].getObjVal());
		}
		
		//output last block 
		if( cbi != -1 && cbj != -1)
			outputBlock(outList, tmpIx, tmpBlock);
		
		_count = 0;
	}

	private static void outputBlock( ArrayList<Pair<Long, FrameBlock>> out, Long key, FrameBlock value ) 
		throws DMLRuntimeException
	{
		//skip output of unassigned blocks
		if( key == -1 )
			return;
		
		//output block
		out.add(new Pair<>(Long.valueOf(key), value));
	}
	
	private static class FrameCell {
		private int iRow;
		private int iCol;
		private Object objVal;
		public int getRow() {
			return iRow;
		}
		public void setRow(int iRow) {
			this.iRow = iRow;
		}
		public int getCol() {
			return iCol;
		}
		public void setCol(int iCol) {
			this.iCol = iCol;
		}
		public Object getObjVal() {
			return objVal;
		}
		public void setObjVal(Object objVal) {
			this.objVal = objVal;
		}
	}
	
	/**
	 * Comparator to sort the reblock buffer by block indexes, where we 
	 * compute the block indexes on-the-fly based on the given cell indexes.
	 * 
	 */
	private static class FrameReblockBufferComparator implements Comparator<FrameCell> {
		@Override
		public int compare(FrameCell arg0, FrameCell arg1) 
		{
			long bi0 = arg0.getRow();
			long bj0 = arg0.getCol();
			long bi1 = arg1.getRow();
			long bj1 = arg1.getCol();
			
			return ( bi0 < bi1 || (bi0 == bi1 && bj0 < bj1) ) ? -1 :
				(( bi0 == bi1 && bj0 == bj1)? 0 : 1);
		}
	}
}
