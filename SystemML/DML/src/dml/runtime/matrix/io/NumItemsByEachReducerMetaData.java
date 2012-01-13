package dml.runtime.matrix.io;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;

public class NumItemsByEachReducerMetaData extends MatrixDimensionsMetaData {
	long[] numItems=null;
	public NumItemsByEachReducerMetaData(MatrixCharacteristics mc, long[] nums)
	{
		super(mc);
		numItems=nums;
	}
	public long[] getNumItemsArray()
	{
		return numItems;
	}
}
