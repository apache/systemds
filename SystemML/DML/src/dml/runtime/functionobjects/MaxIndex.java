package dml.runtime.functionobjects;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class MaxIndex extends IndexFunction{

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
		long maxMatrix=Math.max(in.numRows, in.numColumns);
		int maxBlock=Math.max(in.numRowsPerBlock, in.numColumnsPerBlock);
		out.set(maxMatrix, maxMatrix, maxBlock, maxBlock);
		return false;
	}

}
