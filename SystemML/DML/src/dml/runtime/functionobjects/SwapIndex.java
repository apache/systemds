package dml.runtime.functionobjects;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class SwapIndex extends IndexFunction{

	private static SwapIndex singleObj = null;
	
	private SwapIndex() {
		// nothing to do here
	}
	
	public static SwapIndex getSwapIndexFnObject() {
		if ( singleObj == null )
			singleObj = new SwapIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getColumnIndex(), in.getRowIndex());
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		int temp=in.row;
		out.row=in.column;
		out.column=temp;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(col, row);
		return false;
	}

	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(in.numColumns, in.numRows, in.numColumnsPerBlock, in.numRowsPerBlock);
		return false;
	}
}
