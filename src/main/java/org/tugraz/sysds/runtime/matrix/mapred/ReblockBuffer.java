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


package org.tugraz.sysds.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.SparseBlock.Type;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class ReblockBuffer
{
	public static final int DEFAULT_BUFFER_SIZE = 5000000; //5M x 3x8B = 120MB 

	//buffer <long rowindex, long colindex, long value> (long for sort on flush) 
	private final long[][] _buff;
	private final int _bufflen;
	private int _count;
	private final long _rlen;
	private final long _clen;
	private final int _blen;

	public ReblockBuffer( int buffersize, long rlen, long clen, int blen ) {
		_bufflen = Math.max(buffersize, 16);
		_count = 0;
		_buff = new long[ _bufflen ][3];
		_rlen = rlen;
		_clen = clen;
		_blen = blen;
	}

	public void appendCell( long r, long c, double v ) {
		long tmp = Double.doubleToRawLongBits(v);
		_buff[_count][0] = r;
		_buff[_count][1] = c;
		_buff[_count][2] = tmp;
		_count++;
	}

	public int getSize() {
		return _count;
	}
	
	public int getCapacity() {
		return _bufflen;
	}

	public List<IndexedMatrixValue> flushBufferToBinaryBlocks()
		throws IOException, DMLRuntimeException
	{
		if( _count == 0 )
			return Collections.emptyList();
		
		//Step 1) sort reblock buffer (blockwise, no in-block sorting!)
		Arrays.sort( _buff, 0 ,_count, new ReblockBufferComparator() );
		
		//Step 2) scan for number of created blocks
		long numBlocks = 0; //number of blocks in buffer
		long cbi = -1, cbj = -1; //current block indexes
		for( int i=0; i<_count; i++ )
		{
			long bi = UtilFunctions.computeBlockIndex(_buff[i][0], _blen);
			long bj = UtilFunctions.computeBlockIndex(_buff[i][1], _blen);
			
			//switch to next block
			if( bi != cbi || bj != cbj ) {
				cbi = bi;
				cbj = bj;
				numBlocks++;
			}
		}
		
		//Step 3) output blocks
		ArrayList<IndexedMatrixValue> ret = new ArrayList<>();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(_blen, _blen, _count/numBlocks);
		MatrixIndexes tmpIx = new MatrixIndexes();
		MatrixBlock tmpBlock = new MatrixBlock();
		
		//put values into block and output
		cbi = -1; cbj = -1; //current block indexes
		for( int i=0; i<_count; i++ )
		{
			long bi = UtilFunctions.computeBlockIndex(_buff[i][0], _blen);
			long bj = UtilFunctions.computeBlockIndex(_buff[i][1], _blen);
			
			//output block and switch to next index pair
			if( bi != cbi || bj != cbj ) {
				outputBlock(ret, tmpIx, tmpBlock);
				cbi = bi;
				cbj = bj;
				tmpIx = new MatrixIndexes(bi, bj);
				tmpBlock = new MatrixBlock(
					UtilFunctions.computeBlockSize(_rlen, bi, _blen),
					UtilFunctions.computeBlockSize(_clen, bj, _blen), sparse);
			}
			
			int ci = UtilFunctions.computeCellInBlock(_buff[i][0], _blen);
			int cj = UtilFunctions.computeCellInBlock(_buff[i][1], _blen);
			double tmp = Double.longBitsToDouble(_buff[i][2]);
			tmpBlock.appendValue(ci, cj, tmp); 
		}
		
		//output last block 
		outputBlock(ret, tmpIx, tmpBlock);
		_count = 0;
		return ret;
	}

	private static void outputBlock( ArrayList<IndexedMatrixValue> out, MatrixIndexes key, MatrixBlock value )
		throws IOException, DMLRuntimeException
	{
		//skip output of unassigned blocks
		if( key.getRowIndex() == -1 || key.getColumnIndex() == -1 )
			return;
		
		//sort sparse rows due to blockwise buffer sort and append
		if( value.isInSparseFormat() )
			value.sortSparseRows();
		
		//ensure correct representation (for in-memory blocks)
		value.examSparsity();
	
		//convert ultra-sparse blocks from MCSR to COO in order to 
		//significantly reduce temporary memory pressure until write 
		if( value.isUltraSparse() )
			value = new MatrixBlock(value, Type.COO, false);
		
		//output block
		out.add(new IndexedMatrixValue(key,value));
	}
	
	/**
	 * Comparator to sort the reblock buffer by block indexes, where we
	 * compute the block indexes on-the-fly based on the given cell indexes.
	 * 
	 */
	private class ReblockBufferComparator implements Comparator<long[]> {
		@Override
		public int compare(long[] arg0, long[] arg1) {
			long bi0 = UtilFunctions.computeBlockIndex( arg0[0], _blen );
			long bj0 = UtilFunctions.computeBlockIndex( arg0[1], _blen );
			long bi1 = UtilFunctions.computeBlockIndex( arg1[0], _blen );
			long bj1 = UtilFunctions.computeBlockIndex( arg1[1], _blen );
			return ( bi0 < bi1 || (bi0 == bi1 && bj0 < bj1) ) ? -1 :
				(( bi0 == bi1 && bj0 == bj1)? 0 : 1);
		}
	}
}
