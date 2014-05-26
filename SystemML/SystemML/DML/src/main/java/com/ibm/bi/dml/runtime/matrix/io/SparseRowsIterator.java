/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;

import java.util.Iterator;

/**
 * Iterator for external use of matrix blocks in sparse representation.
 * It allows to linearly iterate only over non-zero values which is
 * important for sparse safe operations.
 * 
 */
public class SparseRowsIterator implements Iterator<IJV>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int rlen = 0;
	private SparseRow[] sparseRows = null;
	private int curRow = -1;
	private int curColIndex = -1;
	private int[] colIndexes = null;
	private double[] values = null;
	private boolean nothingLeft = false;
	private IJV retijv = new IJV();
	
	public SparseRowsIterator(int nrows, SparseRow[] mtx)
	{
		rlen=nrows;
		sparseRows=mtx;
		curRow=0;
		
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
		while(curRow<Math.min(rlen, sparseRows.length) && (sparseRows[curRow]==null || sparseRows[curRow].size()==0))
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
