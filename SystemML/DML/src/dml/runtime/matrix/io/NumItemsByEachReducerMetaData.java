package dml.runtime.matrix.io;

import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;

public class NumItemsByEachReducerMetaData extends MatrixDimensionsMetaData {
	long[] numItems=null;
	int partitionOfZero=-1;
	long numberOfZero=0;
	public NumItemsByEachReducerMetaData(MatrixCharacteristics mc, long[] nums, int part0, long num0)
	{
		super(mc);
		numItems=nums;
		partitionOfZero=part0;
		numberOfZero=num0;
	}
	public NumItemsByEachReducerMetaData(MatrixCharacteristics mc, long[] nums) {
		super(mc);
		numItems=nums;
	}
	public long[] getNumItemsArray()
	{
		return numItems;
	}
	public int getPartitionOfZero()
	{
		return partitionOfZero;
	}
	public long getNumberOfZero()
	{
		return numberOfZero;
	}
}
