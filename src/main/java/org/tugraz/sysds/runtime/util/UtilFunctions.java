/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.tugraz.sysds.runtime.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.SparseBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.Pair;
import org.tugraz.sysds.runtime.matrix.mapred.IndexedMatrixValue;

public class UtilFunctions 
{
	//for accurate cast of double values to int and long 
	//IEEE754: binary64 (double precision) eps = 2^(-53) = 1.11 * 10^(-16)
	//(same epsilon as used for matrix index cast in R)
	public static final double DOUBLE_EPS = Math.pow(2, -53);
	
	//prime numbers for old hash function (divide prime close to max int, 
	//because it determines the max hash domain size
	public static final long ADD_PRIME1 = 99991;
	public static final int DIVIDE_PRIME = 1405695061; 
	
	public static int intHashCode(int key1, int key2) {
		return 31 * (31 + key1) + key2;
	}
	
	public static int longHashCode(long key1) {
		return (int)(key1^(key1>>>32));
	}

	/**
	 * Returns the hash code for a long-long pair. This is the default
	 * hash function for the keys of a distributed matrix in MR/Spark.
	 * 
	 * @param key1 first long key
	 * @param key2 second long key
	 * @return hash code
	 */
	public static int longHashCode(long key1, long key2) {
		//basic hash mixing of two longs hashes (similar to
		//Arrays.hashCode(long[]) but w/o array creation/copy)
		int h = 31 + (int)(key1 ^ (key1 >>> 32));
		return h*31 + (int)(key2 ^ (key2 >>> 32));
	}
	
	/**
	 * Returns the hash code for a long-long-long triple. This is the default
	 * hash function for the keys of a distributed matrix in MR/Spark.
	 * 
	 * @param key1 first long key
	 * @param key2 second long key
	 * @param key3 third long key
	 * @return hash code
	 */
	public static int longHashCode(long key1, long key2, long key3) {
		//basic hash mixing of three longs hashes (similar to
		//Arrays.hashCode(long[]) but w/o array creation/copy)
		int h1 = 31 + (int)(key1 ^ (key1 >>> 32));
		int h2 = h1*31 + (int)(key2 ^ (key2 >>> 32));
		return h2*31 + (int)(key3 ^ (key3 >>> 32));
	}
	
	public static int nextIntPow2( int in ) {
		int expon = (in==0) ? 0 : 32-Integer.numberOfLeadingZeros(in-1);
		long pow2 = pow(2, expon);
		return (int)((pow2>Integer.MAX_VALUE)?Integer.MAX_VALUE : pow2);	
	}
	
	public static long pow(int base, int exp) {
		return (base==2 && 0 <= exp && exp < 63) ?
			1L << exp : (long)Math.pow(base, exp);
	}
	
	/**
	 * Computes the 1-based block index based on the global cell index and block size meta
	 * data. See computeCellIndex for the inverse operation.
	 * 
	 * @param cellIndex global cell index
	 * @param blockSize block size
	 * @return 1-based block index
	 */
	public static long computeBlockIndex(long cellIndex, int blockSize) {
		return (cellIndex-1)/blockSize + 1;
	}
	
	/**
	 * Computes the 0-based cell-in-block index based on the global cell index and block
	 * size meta data. See computeCellIndex for the inverse operation.
	 * 
	 * @param cellIndex global cell index
	 * @param blockSize block size
	 * @return 0-based cell-in-block index
	 */
	public static int computeCellInBlock(long cellIndex, int blockSize) {
		return (int) ((cellIndex-1)%blockSize);
	}
	
	/**
	 * Computes the global 1-based cell index based on the block index, block size meta data,
	 * and specific 0-based in-block cell index.
	 * 
	 * NOTE: this is equivalent to cellIndexCalculation.
	 * 
	 * @param blockIndex block index
	 * @param blockSize block size
	 * @param cellInBlock 0-based cell-in-block index
	 * @return global 1-based cell index
	 */
	public static long computeCellIndex( long blockIndex, int blockSize, int cellInBlock ) {
		return (blockIndex-1)*blockSize + 1 + cellInBlock;
	}
	
	/**
	 * Computes the actual block size based on matrix dimension, block index, and block size
	 * meta data. For boundary blocks, the actual block size is less or equal than the block 
	 * size meta data; otherwise they are identical.  
	 *  
	 * @param len matrix dimension
	 * @param blockIndex block index
	 * @param blockSize block size metadata
	 * @return actual block size
	 */
	public static int computeBlockSize( long len, long blockIndex, long blockSize ) {
		long remain = len - (blockIndex-1)*blockSize;
		return (int)Math.min(blockSize, remain);
	}

	/**
	 * Calculates the number of the block this index refers to (basically a linearisation).
	 * @param ix the dimensional indexes
	 * @param dims length of dimensions
	 * @param blen length of blocks
	 * @return the number of the block
	 */
	public static long computeBlockNumber(int[] ix, long[] dims, int blen) {
		long pos = ix[ix.length - 1] - 1;
		for (int i = ix.length - 2; i >= 0; i--) {
			pos += (ix[i] - 1) * Math.ceil((double)dims[i + 1] / blen);
		}
		return pos;
	}

	public static ArrayList<Integer> getBalancedBlockSizesDefault(int len, int k, boolean constK) {
		int nk = constK ? k : roundToNext(Math.min(8*k,len/32), k);
		return getBalancedBlockSizes(len, nk);
	}
	
	public static ArrayList<Integer> getAlignedBlockSizes(int len, int k, int align) {
		int blklen = (int)(Math.ceil((double)len/k));
		blklen += ((blklen%align != 0) ? align-blklen%align : 0);
		ArrayList<Integer> ret = new ArrayList<>(len/blklen);
		for(int i=0; i<len; i+=blklen)
			ret.add(Math.min(blklen, len-i));
		return ret;
	}
	
	private static ArrayList<Integer> getBalancedBlockSizes(int len, int k) {
		ArrayList<Integer> ret = new ArrayList<>(k);
		int base = len / k;
		int rest = len % k;
		for( int i=0; i<k; i++ ) {
			int val = base + (i<rest?1:0);
			if( val > 0 )
				ret.add(val);
		}
		return ret;
	}
	
	public static boolean isInBlockRange( MatrixIndexes ix, int blen, long rl, long ru, long cl, long cu )
	{
		long bRLowerIndex = (ix.getRowIndex()-1)*blen + 1;
		long bRUpperIndex = ix.getRowIndex()*blen;
		long bCLowerIndex = (ix.getColumnIndex()-1)*blen + 1;
		long bCUpperIndex = ix.getColumnIndex()*blen;
		
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

	public static boolean isInFrameBlockRange( Long ix, int blen, long rl, long ru )
	{
		if(rl > ix+blen-1 || ru < ix)
			return false;
		else
			return true;
	}

	public static boolean isInBlockRange( MatrixIndexes ix, int blen, IndexRange ixrange ) {
		return isInBlockRange(ix, blen, 
			ixrange.rowStart, ixrange.rowEnd, ixrange.colStart, ixrange.colEnd);
	}

	public static boolean isInFrameBlockRange( Long ix, int blen, IndexRange ixrange )
	{
		return isInFrameBlockRange(ix, blen, ixrange.rowStart, ixrange.rowEnd);
	}
	
	// Reused by both MR and Spark for performing zero out
	public static IndexRange getSelectedRangeForZeroOut(IndexedMatrixValue in, int blen, IndexRange indexRange) 
	{
		IndexRange tempRange = new IndexRange(-1, -1, -1, -1);
		
		long topBlockRowIndex=UtilFunctions.computeBlockIndex(indexRange.rowStart, blen);
		int topRowInTopBlock=UtilFunctions.computeCellInBlock(indexRange.rowStart, blen);
		long bottomBlockRowIndex=UtilFunctions.computeBlockIndex(indexRange.rowEnd, blen);
		int bottomRowInBottomBlock=UtilFunctions.computeCellInBlock(indexRange.rowEnd, blen);
		
		long leftBlockColIndex=UtilFunctions.computeBlockIndex(indexRange.colStart, blen);
		int leftColInLeftBlock=UtilFunctions.computeCellInBlock(indexRange.colStart, blen);
		long rightBlockColIndex=UtilFunctions.computeBlockIndex(indexRange.colEnd, blen);
		int rightColInRightBlock=UtilFunctions.computeCellInBlock(indexRange.colEnd, blen);
		
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
	
	// Reused by both MR and Spark for performing zero out
	public static IndexRange getSelectedRangeForZeroOut(Pair<Long, FrameBlock> in, int blen, IndexRange indexRange, long lSrcRowIndex, long lDestRowIndex)  {
		int iRowStart = (indexRange.rowStart <= lDestRowIndex) ?
			0 : (int) (indexRange.rowStart - in.getKey());
		int iRowEnd = (int) Math.min(indexRange.rowEnd - lSrcRowIndex, blen)-1;
		int iColStart = UtilFunctions.computeCellInBlock(indexRange.colStart, blen);
		int iColEnd = UtilFunctions.computeCellInBlock(indexRange.colEnd, blen);
		return  new IndexRange(iRowStart, iRowEnd, iColStart, iColEnd);
	}

	/**
	 * JDK8 floating decimal double parsing, which is generally faster
	 * than &lt;JDK8 parseDouble and works well in multi-threaded tasks.
	 * 
	 * @param str string to parse to double
	 * @return double value
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
	
	public static int toInt( double val ) {
		return (int) (Math.signum(val)
			* Math.floor(Math.abs(val) + DOUBLE_EPS));
	}
	
	public static long toLong( double val ) {
		return (long) (Math.signum(val)
			* Math.floor(Math.abs(val) + DOUBLE_EPS));
	}
	
	public static int toInt(Object obj) {
		return (obj instanceof Long) ?
			((Long)obj).intValue() : ((Integer)obj).intValue();
	}
	
	public static long getSeqLength(double from, double to, double incr) {
		return getSeqLength(from, to, incr, true);
	}
	
	public static long getSeqLength(double from, double to, double incr, boolean check) {
		//Computing the length of a sequence with 1 + floor((to-from)/incr) 
		//can lead to incorrect results due to round-off errors in case of 
		//a very small increment. Hence, we use a different formulation 
		//that exhibits better numerical stability by avoiding the subtraction
		//of numbers of different magnitude.
		if( check && (Double.isNaN(from) || Double.isNaN(to) || Double.isNaN(incr) 
			|| (from > to && incr > 0) || (from < to && incr < 0)) ) {
			throw new RuntimeException("Invalid seq parameters: ("+from+", "+to+", "+incr+")");
		}
		return 1L + (long) Math.floor(to/incr - from/incr);
	}
	
	/**
	 * Obtain sequence list
	 * 
	 * @param low   lower bound (inclusive)
	 * @param up    upper bound (inclusive)
	 * @param incr  increment 
	 * @return list of integers
	 */
	public static List<Integer> getSeqList(int low, int up, int incr) {
		ArrayList<Integer> ret = new ArrayList<>();
		for( int i=low; i<=up; i+=incr )
			ret.add(i);
		return ret;
	}
	
	/**
	 * Obtain sequence array
	 * 
	 * @param low   lower bound (inclusive)
	 * @param up    upper bound (inclusive)
	 * @param incr  increment 
	 * @return array of integers
	 */
	public static int[] getSeqArray(int low, int up, int incr) {
		int len = (int) getSeqLength(low, up, incr);
		int[] ret = new int[len];
		for( int i=0, val=low; i<len; i++, val+=incr )
			ret[i] = val;
		return ret;
	}
	
 	public static int roundToNext(int val, int factor) {
		//round up to next non-zero multiple of factor
		int pval = Math.max(val, factor);
		return ((pval + factor-1) / factor) * factor;
	}

	public static Object doubleToObject(ValueType vt, double in) {
		return doubleToObject(vt, in, true);
	}

	public static Object doubleToObject(ValueType vt, double in, boolean sparse) {
		if( in == 0 && sparse) return null;
		switch( vt ) {
			case STRING:  return String.valueOf(in);
			case BOOLEAN: return (in!=0);
			case INT64:     return UtilFunctions.toLong(in);
			case FP64:  return in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	public static Object stringToObject(ValueType vt, String in) {
		if( in == null )  return null;
		switch( vt ) {
			case STRING:  return in;
			case BOOLEAN: return Boolean.parseBoolean(in);
			case INT64:     return Long.parseLong(in);
			case FP64:  return Double.parseDouble(in);
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	public static double objectToDouble(ValueType vt, Object in) {
		if( in == null )  return 0;
		switch( vt ) {
			case FP64:    return (Double)in;
			case FP32:    return (Float)in;
			case INT64:   return (Long)in;
			case INT32:   return (Integer)in;
			case BOOLEAN: return ((Boolean)in) ? 1 : 0;
			case STRING:  return !((String)in).isEmpty() ? Double.parseDouble((String)in) : 0;
			default: 
				throw new DMLRuntimeException("Unsupported value type: "+vt);
		}
	}

	public static String objectToString( Object in ) {
		return (in !=null) ? in.toString() : null;
	}
	
	/**
	 * Convert object to string
	 * 
	 * @param in object
	 * @param ignoreNull If this flag has set, it will ignore null. This flag is mainly used in merge functionality to override data with "null" data.
	 * @return string representation of object
	 */
	public static String objectToString( Object in, boolean ignoreNull ) {
		String strReturn = objectToString(in); 
		if( strReturn == null )
			return strReturn;
		else if (ignoreNull){
			if(in instanceof Double && ((Double)in).doubleValue() == 0.0)
				return null;
			else if(in instanceof Long && ((Long)in).longValue() == 0)
				return null;
			else if(in instanceof Boolean && ((Boolean)in).booleanValue() == false)
				return null;
			else if(in instanceof String && ((String)in).trim().length() == 0)
				return null;
			else
				return strReturn;
		} 
		else
			return strReturn;
	}

	public static Object objectToObject(ValueType vt, Object in) {
		if( in instanceof Double && vt == ValueType.FP64 
			|| in instanceof Long && vt == ValueType.INT64
			|| in instanceof Boolean && vt == ValueType.BOOLEAN
			|| in instanceof String && vt == ValueType.STRING )
			return in; //quick path to avoid double parsing
		else
			return stringToObject(vt, objectToString(in) );
	}

	public static Object objectToObject(ValueType vt, Object in, boolean ignoreNull ) {
		String str = objectToString(in, ignoreNull);
		if (str==null || vt == ValueType.STRING)
			return str;
		else
			return stringToObject(vt, str); 
	}	

	public static int compareTo(ValueType vt, Object in1, Object in2) {
		if(in1 == null && in2 == null) return 0;
		else if(in1 == null) return -1;
		else if(in2 == null) return 1;
 
		switch( vt ) {
			case STRING:  return ((String)in1).compareTo((String)in2);
			case BOOLEAN: return ((Boolean)in1).compareTo((Boolean)in2);
			case INT64:     return ((Long)in1).compareTo((Long)in2);
			case FP64:  return ((Double)in1).compareTo((Double)in2);
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	/**
	 * Compares two version strings of format x.y.z, where x is major,
	 * y is minor, and z is maintenance release.
	 * 
	 * @param version1 first version string
	 * @param version2 second version string
	 * @return 1 if version1 greater, -1 if version2 greater, 0 if equal
	 */
	public static int compareVersion( String version1, String version2 ) {
		String[] partsv1 = version1.split("\\.");
		String[] partsv2 = version2.split("\\.");
		int len = Math.min(partsv1.length, partsv2.length);
		for( int i=0; i<partsv1.length && i<len; i++ ) {
			Integer iv1 = Integer.parseInt(partsv1[i]);
			Integer iv2 = Integer.parseInt(partsv2[i]);
			if( iv1.compareTo(iv2) != 0 )
				return iv1.compareTo(iv2);
		}		
		return 0; //equal 
	}
	
	public static boolean isIntegerNumber( String str ) {
		byte[] c = str.getBytes();
		for( int i=0; i<c.length; i++ )
			if( c[i] < 48 || c[i] > 57 )
				return false;
		return true;
	}
	
	public static int[] getSortedSampleIndexes(int range, int sampleSize) {
		RandomDataGenerator rng = new RandomDataGenerator();
		int[] sample = rng.nextPermutation(range, sampleSize);
		Arrays.sort(sample);
		return sample;
	}

	public static byte max( byte[] array ) {
		byte ret = Byte.MIN_VALUE;
		for( int i=0; i<array.length; i++ )
			ret = (array[i]>ret)?array[i]:ret;
		return ret;	
	}
	
	public static String unquote(String s) {
		if (s != null && s.length() >=2 
			&& ((s.startsWith("\"") && s.endsWith("\"")) 
			|| (s.startsWith("'") && s.endsWith("'")))) {
			s = s.substring(1, s.length() - 1);
		}
		return s;
	}
	
	public static String quote(String s) {
		return "\"" + s + "\"";
	}

	/**
	 * Parses a memory size with optional g/m/k quantifiers into its
	 * number representation.
	 * 
	 * @param arg memory size as readable string
	 * @return byte count of memory size
	 */
	public static long parseMemorySize(String arg) {
		if ( arg.endsWith("g") || arg.endsWith("G") )
			return Long.parseLong(arg.substring(0,arg.length()-1)) * 1024 * 1024 * 1024;
		else if ( arg.endsWith("m") || arg.endsWith("M") )
			return Long.parseLong(arg.substring(0,arg.length()-1)) * 1024 * 1024;
		else if( arg.endsWith("k") || arg.endsWith("K") )
			return Long.parseLong(arg.substring(0,arg.length()-1)) * 1024;
		else 
			return Long.parseLong(arg.substring(0,arg.length()));
	}
	
	/**
	 * Format a memory size with g/m/k quantifiers into its
	 * number representation.
	 * 
	 * @param arg byte count of memory size
	 * @return memory size as readable string
	 */
	public static String formatMemorySize(long arg) {
		if (arg >= 1024 * 1024 * 1024)
			return String.format("%d GB", arg/(1024*1024*1024));
		else if (arg >= 1024 * 1024)
			return String.format("%d MB", arg/(1024*1024));
		else if (arg >= 1024)
			return String.format("%d KB", arg/(1024));
		else
			return String.format("%d", arg);
	}

	public static double getDouble(Object obj) {
		return (obj instanceof Double) ? (Double)obj :
			Double.parseDouble(obj.toString());
	}

	public static boolean isNonZero(Object obj) {
		if( obj instanceof Double ) 
			return ((Double) obj) != 0;
		else {
			//avoid expensive double parsing
			String sobj = obj.toString();
			return (!sobj.equals("0") && !sobj.equals("0.0"));
		}
	}
	
	public static int computeNnz(double[] a, int ai, int len) {
		int lnnz = 0;
		for( int i=ai; i<ai+len; i++ )
			lnnz += (a[i] != 0) ? 1 : 0;
		return lnnz;
	}

	public static int computeNnz(float[] a, int ai, int len) {
		int lnnz = 0;
		for( int i=ai; i<ai+len; i++ )
			lnnz += (a[i] != 0) ? 1 : 0;
		return lnnz;
	}
	
	public static int computeNnz(long[] a, int ai, int len) {
		int lnnz = 0;
		for( int i=ai; i<ai+len; i++ )
			lnnz += (a[i] != 0) ? 1 : 0;
		return lnnz;
	}
	
	public static int computeNnz(int[] a, int ai, int len) {
		int lnnz = 0;
		for( int i=ai; i<ai+len; i++ )
			lnnz += (a[i] != 0) ? 1 : 0;
		return lnnz;
	}
	
	public static int computeNnz(BitSet a, int ai, int len) {
		int lnnz = 0;
		for( int i=ai; i<ai+len; i++ )
			lnnz += a.get(i) ? 1 : 0;
		return lnnz;
	}

	public static int computeNnz(String[] a, int ai, int len) {
		int lnnz = 0;
		for( int k=ai; k<ai+len; k++ )
			lnnz += (a[k] != null && !a[k].isEmpty() && Double.parseDouble(a[k]) != 0) ? 1 : 0;
		return lnnz;
	}

	public static long computeNnz(SparseBlock a, int[] aix, int ai, int alen) {
		long lnnz = 0;
		for( int k=ai; k<ai+alen; k++ )
			lnnz += a.size(aix[k]);
		return lnnz;
	}

	public static ValueType[] nCopies(int n, ValueType vt) {
		ValueType[] ret = new ValueType[n];
		Arrays.fill(ret, vt);
		return ret;
	}

	public static int frequency(ValueType[] schema, ValueType vt) {
		int count = 0;
		for( ValueType tmp : schema )
			count += tmp.equals(vt) ? 1 : 0;
		return count;
	}

	public static ValueType[] copyOf(ValueType[] schema1, ValueType[] schema2) {
		return (ValueType[]) ArrayUtils.addAll(schema1, schema2);
	}
	
	public static int countNonZeros(double[] data, int pos, int len) {
		int ret = 0;
		for( int i=pos; i<pos+len; i++ )
			ret += (data[i] != 0) ? 1 : 0;
		return ret;
	}
	
	public static boolean containsZero(double[] data, int pos, int len) {
		for( int i=pos; i<pos+len; i++ )
			if( data[i] == 0 )
				return true;
		return false;
	}
	
	@SafeVarargs
	public static <T> List<T> asList(List<T>... inputs) {
		List<T> ret = new ArrayList<>();
		for( List<T> list : inputs )
			ret.addAll(list);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(List<T>... inputs) {
		Set<T> ret = new HashSet<>();
		for( List<T> list : inputs )
			ret.addAll(list);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(T[]... inputs) {
		Set<T> ret = new HashSet<>();
		for( T[] input : inputs )
			for( T element : input )
				ret.add(element);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(T... inputs) {
		Set<T> ret = new HashSet<>();
		for( T element : inputs )
			ret.add(element);
		return ret;
	}
	
	public static <T> Stream<T> getStream(Iterator<T> iter) {
		Iterable<T> iterable = () -> iter;
		return StreamSupport.stream(iterable.spliterator(), false);
	}

	public static long prod(long[] arr) {
		long ret = 1;
		for(int i=0; i<arr.length; i++)
			ret *= arr[i];
		return ret;
	}
	
	public static long prod(int[] arr) {
		long ret = 1;
		for(int i=0; i<arr.length; i++)
			ret *= arr[i];
		return ret;
	}
	
	public static long prod(int[] arr, int off) {
		long ret = 1;
		for(int i=off; i<arr.length; i++)
			ret *= arr[i];
		return ret;
	}

	public static void getBlockBounds(TensorIndexes ix, long[] dims, int blen, int[] lower, int[] upper) {
		for (int i = 0; i < dims.length; i++) {
			lower[i] = (int) (ix.getIndex(i) - 1) * blen;
			upper[i] = (int) (lower[i] + dims[i] - 1);
		}
		upper[upper.length - 1]++;
		for (int i = upper.length - 1; i > 0; i--) {
			if (upper[i] == dims[i]) {
				upper[i] = 0;
				upper[i - 1]++;
			}
			else
				break;
		}
	}
}
