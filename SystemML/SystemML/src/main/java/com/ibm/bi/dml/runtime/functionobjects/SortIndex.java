/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;


import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;

/**
 * This index function is NOT used for actual sorting but just as a reference
 * in ReorgOperator in order to identify sort operations.
 * 
 */
public class SortIndex extends IndexFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int     _col        = -1;
	private boolean _decreasing = false;
	private boolean _ixreturn   = false;
	
	private SortIndex() {
		// nothing to do here
	}

	public static SortIndex getSortIndexFnObject(int col, boolean decreasing, boolean indexreturn) 
	{
		SortIndex ix = new SortIndex();
		ix._col = col;
		ix._decreasing = decreasing;
		ix._ixreturn = indexreturn;
		
		return ix;
	}

	public int getCol() {
		return _col;
	}
	
	public boolean getDecreasing() {
		return _decreasing;
	}
	
	public boolean getIndexReturn() {
		return _ixreturn;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	public boolean computeDimension(int row, int col, CellIndex retDim) 
		throws DMLRuntimeException 
	{
		retDim.set(row, _ixreturn?1:col);
		return false;
	}
	

}
