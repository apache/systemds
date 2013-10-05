/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;


public class OffsetColumnIndex extends IndexFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//private static OffsetColumnIndex singleObj = null;
	private int offset, numRowsInOutput, numColumnsInOutput;
	
	private OffsetColumnIndex(int offset) {
		this.offset = offset;
	}
	
	public static OffsetColumnIndex getOffsetColumnIndexFnObject(int offset) {
		return new OffsetColumnIndex(offset);
		//if ( singleObj == null )
		//	singleObj = new OffsetColumnIndex(offset);
		//return singleObj;
	}
	
	public void setOutputSize(int rows, int columns){
		numRowsInOutput = rows;
		numColumnsInOutput = columns;
	}
	
	public void setOffset(int offset){
		this.offset = offset;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getRowIndex(), in.getColumnIndex()+offset);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		out.row=in.row;
		out.column=offset+in.column;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(numRowsInOutput, numColumnsInOutput);
		return false;
	}

	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(numRowsInOutput, numColumnsInOutput, in.numRowsPerBlock, in.numColumnsPerBlock);
		return false;
	}
}
