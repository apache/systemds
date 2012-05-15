package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class ReduceCol  extends IndexFunction{

	private static ReduceCol singleObj = null;
	
	private ReduceCol() {
		// nothing to do here
	}
	
	public static ReduceCol getReduceColFnObject() {
		if ( singleObj == null )
			singleObj = new ReduceCol();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	/*
	 * NOTE: index starts from 1 for cells in a matrix, but index starts from 0 for cells inside a block
	 */
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getRowIndex(), 1);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		out.row=in.row;
		out.column=0;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(row, 1);
		return true;
	}

	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(in.numRows, 1, in.numRowsPerBlock, 1);
		return true;
	}
}
