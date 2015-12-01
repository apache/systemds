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

import java.util.Iterator;

/**
 * Iterator for external use of matrix blocks in sparse representation.
 * It allows to linearly iterate only over non-zero values which is
 * important for sparse safe operations.
 * 
 */
public class SparseRowsIterator implements Iterator<IJV>
{
	
	private int rlen = 0;
	private SparseRow[] sparseRows = null;
	private int curRow = -1;
	private int curColIndex = -1;
	private int[] colIndexes = null;
	private double[] values = null;
	private boolean nothingLeft = false;
	private IJV retijv = new IJV();

	//allow initialization from package or subclasses
	protected SparseRowsIterator(int nrows, SparseRow[] mtx)
	{
		rlen=nrows;
		sparseRows=mtx;
		curRow=0;
		
		if(sparseRows==null)
			nothingLeft=true;
		else
			findNextNonZeroRow();
	}
	
	//allow initialization from package or subclasses
	protected SparseRowsIterator(int currow, int nrows, SparseRow[] mtx)
	{
		rlen=nrows;
		sparseRows=mtx;
		curRow=currow;
		
		if(sparseRows==null)
			nothingLeft=true;
		else
			findNextNonZeroRow();
	}
	
	@Override
	public boolean hasNext() {
		if(nothingLeft)
			return false;
		else
			return true;
	}

	@Override
	public IJV next( ) {
		retijv.set(curRow, colIndexes[curColIndex], values[curColIndex]);
		curColIndex++;
		if(curColIndex>=sparseRows[curRow].size())
		{
			curRow++;
			findNextNonZeroRow();
		}
		return retijv;
	}

	@Override
	public void remove() {
		throw new RuntimeException("SparseCellIterator.remove should not be called!");
		
	}		
	
	/**
	 * 
	 */
	private void findNextNonZeroRow() 
	{
		while(curRow<Math.min(rlen, sparseRows.length) && (sparseRows[curRow]==null || sparseRows[curRow].isEmpty()))
			curRow++;
		if(curRow>=Math.min(rlen, sparseRows.length))
			nothingLeft=true;
		else
		{
			curColIndex=0;
			colIndexes=sparseRows[curRow].getIndexContainer();
			values=sparseRows[curRow].getValueContainer();
		}
	}
}
