/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;

public class NumItemsByEachReducerMetaData extends MatrixDimensionsMetaData 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long[] numItems=null;
	private int partitionOfZero=-1;
	private long numberOfZero=0;
	
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
	
	@Override
	public Object clone()
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(matchar);
		NumItemsByEachReducerMetaData ret = new NumItemsByEachReducerMetaData(mc, numItems, partitionOfZero, numberOfZero);
	
		return ret;
	}
}
