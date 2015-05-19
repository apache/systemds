/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;


public class MaxIndex extends IndexFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -4941564912238185729L;

	private static MaxIndex singleObj = null;
	
	private MaxIndex() {
		// nothing to do here
	}
	
	public static MaxIndex getMaxIndexFnObject() {
		if ( singleObj == null )
			singleObj = new MaxIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		long max = Math.max(in.getRowIndex(), in.getColumnIndex());
		out.setIndexes(max, max);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		int max = Math.max(in.row, in.column);
		out.set(max, max);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		int max=Math.max(row, col);
		retDim.set(max, max);
		return false;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		long maxMatrix=Math.max(in.getRows(), in.getCols());
		int maxBlock=Math.max(in.getRowsPerBlock(), in.getColsPerBlock());
		out.set(maxMatrix, maxMatrix, maxBlock, maxBlock);
		return false;
	}

}
