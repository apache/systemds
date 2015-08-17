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

package com.ibm.bi.dml.runtime.matrix.data;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Ctable map is an abstraction for the hashmap used for ctable's hash group-by
 * because this structure is passed through various interfaces. This makes it 
 * easier to (1) exchange the underlying data structure and (2) maintain statistics 
 * like max row/column in order to prevent scans during data conversion.
 * 
 */
public class CTableMap 
{
	
	private HashMap<MatrixIndexes, Double> _map = null;
	private long _maxRow = -1;
	private long _maxCol = -1;
	
	public CTableMap() {
		_map = new HashMap<MatrixIndexes, Double>();
		_maxRow = -1;
		_maxCol = -1;
	}
	
	/**
	 * 
	 * @return
	 */
	public int size() 
	{
		return _map.size();
	}
	
	/**
	 * 
	 * @return
	 */
	@Deprecated
	public Set<Entry<MatrixIndexes, Double>> entrySet()
	{
		return _map.entrySet();
	}
	
	/**
	 * 
	 * @return
	 */
	public long getMaxRow() {
		return _maxRow;
	}
	
	/**
	 * 
	 * @return
	 */
	public long getMaxColumn() {
		return _maxCol;
	}

	/**
	 * 
	 * @param row
	 * @param key
	 * @return
	 */
	public double get( long row, long col )
	{
		MatrixIndexes key = new MatrixIndexes(row, col);		
		return _map.get(key);
	}
	
	/**
	 * 
	 * @param row
	 * @param col
	 * @param w
	 */
	public void aggregate(long row, long col, double w) 
	{
		//hash group-by for core ctable computation
		MatrixIndexes key = new MatrixIndexes(row, col);		
		Double oldval = _map.get(key);
		if( oldval != null ) //existing group
			_map.put(key, oldval+w);
		else             //non-existing group 
			_map.put(key, w);
		
		//maintain internal summaries 
		_maxRow = Math.max(_maxRow, row);
		_maxCol = Math.max(_maxCol, col);
	}
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @return
	 */
	public MatrixBlock toMatrixBlock(int rlen, int clen)
	{
		//allocate new matrix block
		int nnz = _map.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz); 		
		MatrixBlock mb = new MatrixBlock(rlen, clen, sparse, nnz);
		
		// copy map values into new matrix block
		if( sparse ) //SPARSE <- cells
		{
			//append cells to sparse target (prevent shifting)
			for( Entry<MatrixIndexes,Double> e : _map.entrySet() ) 
			{
				MatrixIndexes index = e.getKey();
				double value = e.getValue();
				int rix = (int)index.getRowIndex();
				int cix = (int)index.getColumnIndex();
				if( value != 0 && rix<=rlen && cix<=clen )
					mb.appendValue( rix-1, cix-1, value );
			}
			
			//sort sparse target representation
			mb.sortSparseRows();
		}
		else  //DENSE <- cells
		{
			//directly insert cells into dense target 
			for( Entry<MatrixIndexes,Double> e : _map.entrySet() ) 
			{
				MatrixIndexes index = e.getKey();
				double value = e.getValue();
				int rix = (int)index.getRowIndex();
				int cix = (int)index.getColumnIndex();
				if( value != 0 && rix<=rlen && cix<=clen )
					mb.quickSetValue( rix-1, cix-1, value );
			}
		}
		
		return mb;
	}
}
