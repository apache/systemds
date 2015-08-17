/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;

public class UtilFunctions 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		

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
	
	/**
	 * 
	 * @param len
	 * @param blockIndex
	 * @param blockSize
	 * @return
	 */
	public static int computeBlockSize( long len, long blockIndex, long blockSize )
	{
		long remain = len - (blockIndex-1)*blockSize;
		return (int)Math.min(blockSize, remain);
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
	
	/**
	 * 
	 * @param ix
	 * @param brlen
	 * @param bclen
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @return
	 */
	public static boolean isInBlockRange( MatrixIndexes ix, int brlen, int bclen, long rl, long ru, long cl, long cu )
	{
		long bRLowerIndex = (ix.getRowIndex()-1)*brlen + 1;
		long bRUpperIndex = ix.getRowIndex()*brlen;
		long bCLowerIndex = (ix.getColumnIndex()-1)*bclen + 1;
		long bCUpperIndex = ix.getColumnIndex()*bclen;
		
		if(rl > bRUpperIndex || ru < bRLowerIndex) {
			return false;
		}
		else if(cl > bCUpperIndex || cu < bCLowerIndex) {
			return false;
		}
		else {
			return true;
		}
	}
	
	// Reused by both MR and Spark for performing zero out
	public static IndexRange getSelectedRangeForZeroOut(IndexedMatrixValue in, int blockRowFactor, int blockColFactor, IndexRange indexRange) 
	{
		IndexRange tempRange = new IndexRange(-1, -1, -1, -1);
		
		long topBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowStart, blockRowFactor);
		int topRowInTopBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
		long bottomBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowEnd, blockRowFactor);
		int bottomRowInBottomBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowEnd, blockRowFactor);
		
		long leftBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colStart, blockColFactor);
		int leftColInLeftBlock=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
		long rightBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colEnd, blockColFactor);
		int rightColInRightBlock=UtilFunctions.cellInBlockCalculation(indexRange.colEnd, blockColFactor);
		
		//no overlap
		if(in.getIndexes().getRowIndex()<topBlockRowIndex || in.getIndexes().getRowIndex()>bottomBlockRowIndex
		   || in.getIndexes().getColumnIndex()<leftBlockColIndex || in.getIndexes().getColumnIndex()>rightBlockColIndex)
		{
			tempRange.set(-1,-1,-1,-1);
			return tempRange;
		}
		
		//get the index range inside the block
		tempRange.set(0, in.getValue().getNumRows()-1, 0, in.getValue().getNumColumns()-1);
		if(topBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowStart=topRowInTopBlock;
		if(bottomBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowEnd=bottomRowInBottomBlock;
		if(leftBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colStart=leftColInLeftBlock;
		if(rightBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colEnd=rightColInRightBlock;
		
		return tempRange;
	}
	
	public static long getTotalLength(NumItemsByEachReducerMetaData metadata) {
		long[] counts=metadata.getNumItemsArray();
		long total=0;
		for(long count: counts)
			total+=count;
		return total;
	}
	
	public static long getLengthForInterQuantile(NumItemsByEachReducerMetaData metadata, double p)
	{
		long total = UtilFunctions.getTotalLength(metadata);
		long lpos=(long)Math.ceil(total*p);//lower bound is inclusive
		long upos=(long)Math.ceil(total*(1-p));//upper bound is inclusive
		//System.out.println("getLengthForInterQuantile(): " + (upos-lpos+1));
		return upos-lpos+1;
	}

	/**
	 * JDK8 floating decimal double parsing, which is generally faster
	 * than <JDK8 parseDouble and works well in multi-threaded tasks.
	 * 
	 * @param str
	 * @return
	 */
	public static double parseToDouble(String str)
	{
		//return FloatingDecimal.parseDouble(str);
    	return Double.parseDouble(str);
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
	
	public static boolean isIntegerNumber( String str )
	{
		byte[] c = str.getBytes();
		for( int i=0; i<c.length; i++ )
			if( c[i] < 48 || c[i] > 57 )
				return false;
		return true;
	}
	
	public static boolean isSimpleDoubleNumber( String str )
	{
		//true if all chars numeric or - or .
		byte[] c = str.getBytes();
		for( int i=0; i<c.length; i++ )
			if( (c[i] < 48 || c[i] > 57) && !(c[i]==45 || c[i]==46) )
				return false;
		return true;
	}
	
	public static byte max( byte[] array )
	{
		byte ret = Byte.MIN_VALUE;
		for( int i=0; i<array.length; i++ )
			ret = (array[i]>ret)?array[i]:ret;
		return ret;	
	}
	
	public static String unquote(String s) {
		if (s != null
				&& ((s.startsWith("\"") && s.endsWith("\"")) 
					|| (s.startsWith("'") && s.endsWith("'")))) {
			s = s.substring(1, s.length() - 1);
		}
		return s;
	}
	
	public static String quote(String s) {
		return "\"" + s + "\"";
	}

	public static String toString(int[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append(list[0]);
		for(int i=1; i<list.length; i++) {
			sb.append(",");
			sb.append(list[i]);
		}
		return sb.toString();
	}
}
