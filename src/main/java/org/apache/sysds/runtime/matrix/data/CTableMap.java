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

package org.apache.sysds.runtime.matrix.data;

import java.util.Iterator;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap.ADoubleEntry;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap.EntryType;

/**
 * Ctable map is an abstraction for the hashmap used for ctable's hash group-by
 * because this structure is passed through various interfaces. This makes it 
 * easier to (1) exchange the underlying data structure and (2) maintain statistics 
 * like max row/column in order to prevent scans during data conversion.
 * 
 */
public class CTableMap 
{
	private final LongLongDoubleHashMap _map;
	private long _maxRow = -1;
	private long _maxCol = -1;
	
	public CTableMap() {
		this(EntryType.LONG);
	}

	public CTableMap(EntryType type) {
		_map = new LongLongDoubleHashMap(type);
		_maxRow = -1;
		_maxCol = -1;
	}
	
	public int size() {
		return _map.size();
	}
	
	public Iterator<ADoubleEntry> getIterator() {
		return _map.getIterator();
	}

	public long getMaxRow() {
		return _maxRow;
	}

	public long getMaxColumn() {
		return _maxCol;
	}

	public void aggregate(long row, long col, double w) 
	{
		//hash group-by for core ctable computation
		_map.addValue(row, col, w);
		
		//maintain internal summaries 
		_maxRow = Math.max(_maxRow, row);
		_maxCol = Math.max(_maxCol, col);
	}

	public MatrixBlock toMatrixBlock(int rlen, int clen)
	{
		//allocate new matrix block
		int nnz = _map.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz); 
		MatrixBlock mb = new MatrixBlock(rlen, clen, sparse, nnz).allocateBlock();
		
		// copy map values into new matrix block
		if( sparse ) //SPARSE <- cells
		{
			//append cells to sparse target (unordered to avoid shifting)
			SparseBlock sblock = mb.getSparseBlock();
			Iterator<ADoubleEntry> iter2 = _map.getIterator();
			while( iter2.hasNext() ) {
				ADoubleEntry e = iter2.next();
				double value = e.value;
				int rix = (int)e.getKey1();
				int cix = (int)e.getKey2();
				if( value != 0 && rix<=rlen && cix<=clen ) {
					sblock.allocate(rix-1, Math.max(nnz/rlen,1));
					sblock.append( rix-1, cix-1, value );
				}
			}
			
			//sort sparse target representation
			mb.sortSparseRows();
			mb.recomputeNonZeros();
		}
		else  //DENSE <- cells
		{
			//directly insert cells into dense target 
			Iterator<ADoubleEntry> iter = _map.getIterator();
			while( iter.hasNext() ) {
				ADoubleEntry e = iter.next();
				double value = e.value;
				int rix = (int)e.getKey1();
				int cix = (int)e.getKey2();
				if( value != 0 && rix<=rlen && cix<=clen )
					mb.quickSetValue( rix-1, cix-1, value );
			}
		}
		
		return mb;
	}
}
