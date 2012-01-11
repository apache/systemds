package dml.runtime.functionobjects;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class ReduceAll extends IndexFunction{

	private static ReduceAll singleObj = null;
	
	private ReduceAll() {
		// nothing to do here
	}
	
	public static ReduceAll getReduceAllFnObject() {
		if ( singleObj == null )
			singleObj = new ReduceAll();
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
		out.setIndexes(1, 1);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		out.set(0, 0);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(1, 1);
		return true;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(1, 1, 1, 1);
		return true;
	}
}
