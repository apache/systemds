package com.ibm.bi.dml.runtime.util;

import java.io.File;

import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class UtilFunctions {

	//for accurate cast of double values to int and long 
	//IEEE754: binary64 (double precision) eps = 2^(-53) = 1.11 * 10^(-16)
	//(same epsilon as used for matrix index cast in R)
	public static double DOUBLE_EPS = Math.pow(2, -53);
	
	
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

	public static int parseToInt( String str )
	{
		int ret = -1;
		if( str.contains(".") )
			ret = toInt( Double.parseDouble(str) );
		else
			ret = Integer.parseInt(str);
		return ret;
	}
	
	public static long parseToLong( String str )
	{
		long ret = -1;
		if( str.contains(".") )
			ret = toLong( Double.parseDouble(str) );
		else
			ret = Long.parseLong(str);
		return ret;
	}
	
	public static int toInt( double val )
	{
		return (int) Math.floor( val + DOUBLE_EPS );
	}
	
	public static long toLong( double val )
	{
		return (long) Math.floor( val + DOUBLE_EPS );
	}
	
	public static void createLocalFileIfNotExist( String dir, String permission )
	{
		File fdir = new File(dir);
		if( !fdir.exists() )
		{
			fdir.mkdirs();
			UtilFunctions.setLocalFilePermissions(fdir, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		}
	}
	
	public static void setLocalFilePermissions( File file, String permissions )
	{
		//note: user and group treated the same way
		char[] c = permissions.toCharArray();
		short sU = (short)(c[0]-48);
		short sO = (short)(c[2]-48); 
		
		file.setExecutable( (sU&1)==1, (sO&1)==0 );
		file.setWritable(   (sU&2)==2, (sO&2)==0 );
		file.setReadable(   (sU&4)==4, (sO&4)==0 );
	}
}
