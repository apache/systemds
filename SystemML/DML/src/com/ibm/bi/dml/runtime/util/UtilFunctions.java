package com.ibm.bi.dml.runtime.util;

import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;

public class UtilFunctions {

	public static int longHashFunc(long v)
	{
		return (int)(v^(v>>>32));
	}
	
	//return one block index given the index in cell format and block size
	public static long blockIndexCalculation(long cellIndex, int blockSize)
	{
		if(cellIndex>0)
			return (cellIndex-1)/blockSize+1;
		else
			return (long)Math.floor((double)(cellIndex-1)/(double)blockSize)+1;
	}
	
	//return cell index in the block, for given index in the cell format and block size
	public static int cellInBlockCalculation(long cellIndex, int blockSize)
	{
		if(cellIndex>0)
			return (int) ((cellIndex-1)%blockSize);
		else
			//return (int) Math.abs((cellIndex-1)%blockSize);
			return (int) ((cellIndex-1)%blockSize)+blockSize;
	}
	
	//given block index and block size and cells in block, return the index in cell format
	public static long cellIndexCalculation(long blockIndex, int blockSize, int cellInBlock)
	{
		return (blockIndex-1)*blockSize+1+cellInBlock;
	}
	
	//all boundaries are inclusive
	public static boolean isOverlap(long s1, long f1, long s2, long f2)
	{
		return !(f2<s1 || f1<s2);
	}
	
	public static boolean isIn(long point, long s, long f)
	{
		return (point>=s && point<=f);
	}
	
	public static long getLengthForInterQuantile(NumItemsByEachReducerMetaData metadata, double p)
	{
		long[] counts=metadata.getNumItemsArray();
		long total=0;
		for(long count: counts)
			total+=count;
		long lpos=(long)Math.ceil(total*p)+1;//lower bound is inclusive
		long upos=(long)Math.ceil(total*(1-p))+1;//upper bound is non inclusive
		return upos-lpos;
	}
}
