package dml.runtime.functionobjects;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class OffsetColumnIndex extends IndexFunction{
	private static OffsetColumnIndex singleObj = null;
	private int offset, numRowsInOutput, numColumnsInOutput;
	
	private OffsetColumnIndex(int offset) {
		this.offset = offset;
	}
	
	public static OffsetColumnIndex getOffsetColumnIndexFnObject(int offset) {
		if ( singleObj == null )
			singleObj = new OffsetColumnIndex(offset);
		return singleObj;
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
