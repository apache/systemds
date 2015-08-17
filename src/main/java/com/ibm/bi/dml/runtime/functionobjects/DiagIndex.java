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


public class DiagIndex extends IndexFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -5294771266108903886L;

	private static DiagIndex singleObj = null;
	
	private DiagIndex() {
		// nothing to do here
	}
	
	public static DiagIndex getDiagIndexFnObject() {
		if ( singleObj == null )
			singleObj = new DiagIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		//only used for V2M
		out.setIndexes(in.getRowIndex(), in.getRowIndex());
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		//only used for V2M
		out.set(in.row, in.row);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		if( col == 1 ) //diagV2M
			retDim.set(row, row);
		else //diagM2V
			retDim.set(row, 1);
		return false;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		if( in.getCols() == 1 ) //diagV2M
			out.set(in.getRows(), in.getRows(), in.getRowsPerBlock(), in.getRowsPerBlock());
		else //diagM2V
			out.set(in.getRows(), 1, in.getRowsPerBlock(), in.getRowsPerBlock());
		return false;
	}

}
