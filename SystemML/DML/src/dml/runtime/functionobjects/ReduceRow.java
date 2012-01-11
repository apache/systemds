package dml.runtime.functionobjects;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class ReduceRow extends IndexFunction{

	private static ReduceRow singleObj = null;
	
	private ReduceRow() {
		// nothing to do here
	}
	
	public static ReduceRow getReduceRowFnObject() {
		if ( singleObj == null )
			singleObj = new ReduceRow();
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
		out.setIndexes(1, in.getColumnIndex());
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		out.row=0;
		out.column=in.column;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(1, col);
		return true;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(1, in.numColumns, 1, in.numColumnsPerBlock);
		return true;
	}

}
