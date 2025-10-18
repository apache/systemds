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

package org.apache.sysds.runtime.util;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TimeZone;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.commons.lang3.time.DateUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.CharArray;
import org.apache.sysds.runtime.frame.data.columns.HashIntegerArray;
import org.apache.sysds.runtime.frame.data.columns.HashLongArray;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class UtilFunctions {
	protected static final Log LOG = LogFactory.getLog(UtilFunctions.class.getName());
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int vLen = SPECIES.length();

	
	private UtilFunctions(){
		// empty private constructor
		// making all calls static
	}

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
	
	public static int intHashCodeRobust(int key1, int key2) {
		// handle overflows to avoid systematic hash code repetitions
		// in long recursive hash computations w/ repeated structure
		long tmp = 31L * (31L + key1) + key2;
		return (tmp < Integer.MAX_VALUE) ?
			(int) tmp : longHashCode(tmp);
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
	 * Computes the next tensor indexes array.
	 * @param tc the tensor characteristics
	 * @param ix the tensor indexes array (will be changed)
	 * @return the tensor indexes array (changed)
	 */
	public static long[] computeNextTensorIndexes(TensorCharacteristics tc, long[] ix) {
		ix[tc.getNumDims() - 1]++;
		for (int i = tc.getNumDims() - 1; i > 0; i--) {
			if (ix[i] == tc.getNumBlocks(i) + 1) {
				ix[i] = 1;
				ix[i - 1]++;
			}
			else {
				break;
			}
		}
		return ix;
	}
	
	/**
	 * Computes the tensor indexes array given a blockIndex we ant to compute. Note that if a sequence of tensor indexes
	 * array will be computed, it is faster to use
	 * <code>UtilFunctions.computeNextTensorIndexes(TensorCharacteristics,long[])</code>.
	 * @param tc the tensor characteristics
	 * @param blockIndex the number of the block ([0-<code>tc.getNumBlocks()</code>[ valid)
	 * @return the tensor index array
	 */
	public static long[] computeTensorIndexes(TensorCharacteristics tc, long blockIndex) {
		long[] ix = new long[tc.getNumDims()];
		for (int j = tc.getNumDims() - 1; j >= 0; j--) {
			ix[j] = 1 + (blockIndex % tc.getNumBlocks(j));
			blockIndex /= tc.getNumBlocks(j);
		}
		return ix;
	}
	
	/**
	 * Computes the slice dimensions and offsets for the block slice of another tensor with the size given by
	 * <code>TensorCharacteristics</code>.
	 * @param tc tensor characteristics of the block to slice
	 * @param blockIx the tensor block index
	 * @param outDims the slice dimension size
	 * @param offset the offset where the slice should start
	 */
	public static void computeSliceInfo(TensorCharacteristics tc, long[] blockIx, int[] outDims,
			int[] offset) {
		for (int i = tc.getNumDims() - 1; i >= 0; i--) {
			outDims[i] = UtilFunctions.computeBlockSize(tc.getDim(i), blockIx[i], tc.getBlocksize());
			offset[i] = (int) ((blockIx[i] - 1) * tc.getBlocksize());
		}
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

	public static List<Pair<Integer,Integer>> getTaskRangesDefault(int len, int k) {
		List<Pair<Integer,Integer>> ret = new ArrayList<>();
		int nk = roundToNext(Math.min(8*k,len/32), k);
		int beg = 0;
		for(Integer blen : getBalancedBlockSizes(len, nk)) {
			ret.add(new Pair<>(beg, beg+blen)); 
			beg = beg+blen;
		}
		return ret;
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
	 * Safe double parsing including handling of NAs. Previously, we also
	 * used this wrapper for handling thread contention in multi-threaded
	 * environments because Double.parseDouble relied on a synchronized cache
	 * (which was replaced with thread-local caches in JDK8).
	 * 
	 * @param str   string to parse to double
	 * @param isNan collection of Nan string which if encountered should be parsed to nan value
	 * @return double value
	 */
	public static double parseToDouble(String str, Set<String> isNan ) {
		return isNan != null && isNan.contains(str) ?
			Double.NaN :
			Double.parseDouble(str);
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
		//Additionally we check the resulting length and add 1 if this check
		//allows inferring that round-off errors happened.
		if( (isSpecial(from) || isSpecial(to) || isSpecial(incr) 
			|| (from > to && incr > 0) || (from < to && incr < 0)) ) {
			if( check )
				throw new RuntimeException("Invalid seq parameters: ("+from+", "+to+", "+incr+")");
			else
				return 0; // invalid loop configuration
		}
		long tmp = (long) Math.floor(to/incr - from/incr);
		if( incr > 0 )
			return 1L + tmp + ((from+(tmp+1)*incr <= to) ? 1 : 0);
		else
			return 1L + tmp + ((from+(tmp+1)*incr >= to) ? 1 : 0);
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
		if( Double.isNaN(in) && sparse) return null;
		switch( vt ) {
			case STRING:  return String.valueOf(in);
			case BOOLEAN: return (in!=0);
			case INT32:   return UtilFunctions.toInt(in);
			case INT64:   return UtilFunctions.toLong(in);
			case FP32:    return ((float)in);
			case FP64:    return in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	public static Object stringToObject(ValueType vt, String in) {
		if( in == null || in.isEmpty() )  return null;
		switch( vt ) {
			case STRING:    return in;
			case BOOLEAN:   return Boolean.parseBoolean(in);
			case UINT4:
			case UINT8:
			case INT32:     return Integer.parseInt(in);
			case INT64:     return Long.parseLong(in);
			case FP64:      return Double.parseDouble(in);
			case FP32:      return Float.parseFloat(in);
			case CHARACTER: return CharArray.parseChar(in);
			case HASH64:    return HashLongArray.parseHashLong(in);
			case HASH32:    return HashIntegerArray.parseHashInt(in);
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	public static double objectToDoubleSafe(ValueType vt, Object in) {
		if(vt == ValueType.STRING && in == null)
			return 0.0;
		if(vt == ValueType.STRING && !NumberUtils.isCreatable((String) in)) {
			return 1.0;
		} else return objectToDouble(vt, in);
	}

	public static double objectToDouble(ValueType vt, Object in) {
		if( in == null )  return Double.NaN;
		switch( vt ) {
			case FP64:    return (Double)in;
			case FP32:    return (Float)in;
			case INT64:   return (Long)in;
			case INT32:   return (Integer)in;
			case BOOLEAN: return ((Boolean)in) ? 1 : 0;
			case CHARACTER: return (Character)in;
			case STRING:
				String inStr = (String) in;
				try {
					return !(inStr).isEmpty() ? Double.parseDouble(inStr) : 0;
				}
				catch(NumberFormatException e) {
					final int len = inStr.length();
					if(len == 1 && inStr.equalsIgnoreCase("T"))
						return 1.0;
					else if (len == 1 && inStr.equalsIgnoreCase("F"))
						return 0.0;
					else if(inStr.equalsIgnoreCase("true"))
						return 1.0;
					else if(inStr.equalsIgnoreCase("false"))
						return 0.0;
					else
						throw new DMLRuntimeException("failed parsing object to double",e);
				}
			default:
				throw new DMLRuntimeException("Unsupported value type: "+vt);
		}
	}

	public static float objectToFloat(ValueType vt, Object in) {
		if(in == null)
			return Float.NaN;
		switch(vt) {
			case FP64:
				return ((Double) in).floatValue();
			case FP32:
				return (Float) in;
			case INT64:
				return (Long) in;
			case INT32:
				return (Integer) in;
			case BOOLEAN:
				return ((Boolean) in) ? 1 : 0;
			case STRING:
				return !((String) in).isEmpty() ? Float.parseFloat((String) in) : 0;
			default:
				throw new DMLRuntimeException("Unsupported value type: " + vt);
		}
	}

	public static char objectToCharacter(ValueType vt, Object in){
		if(in == null)
			return 0;
		switch(vt) {
			case FP64:
				return (char)((Double)in).intValue();
			case FP32:
				return (char)((Float)in).intValue();
			case INT64:
				return (char)((Long)in).longValue();
			case INT32:
				return (char)((Integer)in).intValue();
			case BOOLEAN:
				return ((Boolean) in) ? (char)1 : (char)0;
			case STRING:
				return !((String) in).isEmpty() ? ((String)in).charAt(0) : 0;
			default:
				throw new DMLRuntimeException("Unsupported value type: " + vt);
		}
	}

	public static int objectToInteger(ValueType vt, Object in) {
		if(in == null)
			return 0;
		switch(vt) {
			case FP64:
				return ((Double) in).intValue();
			case FP32:
				return ((Float) in).intValue();
			case INT64:
				return ((Long) in).intValue();
			case INT32:
				return (Integer) in;
			case BOOLEAN:
				return ((Boolean) in) ? 1 : 0;
			case STRING:
				return !((String) in).isEmpty() ? Integer.parseInt((String) in) : 0;
			default:
				throw new DMLRuntimeException("Unsupported value type: " + vt);
		}
	}

	public static long objectToLong(ValueType vt, Object in) {
		if(in == null)
			return 0;
		switch(vt) {
			case FP64:
				return ((Double) in).longValue();
			case FP32:
				return ((Float) in).longValue();
			case INT64:
				return (Long) in;
			case INT32:
				return (Integer) in;
			case BOOLEAN:
				return ((Boolean) in) ? 1 : 0;
			case STRING:
				return !((String) in).isEmpty() ? Long.parseLong((String) in) : 0;
			default:
				throw new DMLRuntimeException("Unsupported value type: " + vt);
		}
	}

	public static boolean objectToBoolean(ValueType vt, Object in) {
		if(in == null)
			return false;
		switch(vt) {
			case FP64:
				return ((Double) in) == 1.0;
			case FP32:
				return ((Float) in) == 1.0;
			case INT64:
				return (Long) in == 1;
			case INT32:
				return (Integer) in == 1;
			case BOOLEAN:
				return ((Boolean) in);
			case STRING:
				return Boolean.parseBoolean((String) in);
			default:
				throw new DMLRuntimeException("Unsupported value type: " + vt);
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
			else if(in instanceof Long && ((Integer)in).intValue() == 0)
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
			|| in instanceof Float && vt == ValueType.FP32
			|| in instanceof Long && (vt == ValueType.INT64 || vt == ValueType.HASH64)
			|| in instanceof Integer && (vt == ValueType.INT32 || vt == ValueType.HASH32)
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
			case INT64:   return ((Long)in1).compareTo((Long)in2);
			case INT32:   return ((Integer)in1).compareTo((Integer)in2);
			case FP64:    return ((Double)in1).compareTo((Double)in2);
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

	public static boolean isBoolean(String str) {
		return String.valueOf(true).equalsIgnoreCase(str) || String.valueOf(false).equalsIgnoreCase(str);
	}
	
	public static boolean isIntegerNumber( String str ) {
		byte[] c = str.getBytes();
		for( int i=0; i<c.length; i++ )
			if( c[i] < 48 || c[i] > 57 )
				return false;
		return true;
	}
	
	public static boolean isSpecial(double value) {
		return Double.isNaN(value) || Double.isInfinite(value);
	}
	
	public static int[] getSortedSampleIndexes(int range, int sampleSize) {
		return getSortedSampleIndexes(range, sampleSize, -1);
	}

	public static int[] getSortedSampleIndexes(int range, int sampleSize, long seed) {
		RandomDataGenerator rng = new RandomDataGenerator();
		if (seed != -1){
			rng.reSeed(seed);
		}
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

	public static int getAsciiAtIdx(String s, int idx) {
		int strlen = s.length();
		int c = 0;
		int javaIdx = idx - 1;
		if (javaIdx >= 0 && javaIdx < strlen) {
			c = s.charAt(javaIdx);
		}
		return c;
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
	
	public static int computeNnz(final double[] a, final int ai, final int len) {
		final int end = ai + len;
		final int rest = (end - ai) % vLen;
		int lnnz = len;

		//start from len and subtract number of zeros because
		//DoubleVector defines an eq but no neq operation
		for(int i = ai; i < ai + rest; i++)
			lnnz -= (a[i] == 0.0) ? 1 : 0;
		for(int i = ai + rest; i < end; i += vLen) {
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, i);
			lnnz -= aVec.eq(0).trueCount();
		}
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
		return ArrayUtils.addAll(schema1, schema2);
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
	
	protected static final Map<String, String> DATE_FORMATS = new HashMap<>() {
		private static final long serialVersionUID = 6826162458614520846L; {
		put("^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$", "yyyy-MM-dd HH:mm:ss");
		put("^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd-MM-yyyy HH:mm:ss");
		put("^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "MM/dd/yyyy HH:mm:ss");
		put("^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$", "yyyy/MM/dd HH:mm:ss");
		put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd MMM yyyy HH:mm:ss");
		put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$", "dd MMMM yyyy HH:mm:ss");
		put("^\\d{8}$", "yyyyMMdd");
		put("^\\d{1,2}-\\d{1,2}-\\d{4}$", "dd-MM-yyyy");
		put("^\\d{4}-\\d{1,2}-\\d{1,2}$", "yyyy-MM-dd");
		put("^\\d{1,2}/\\d{1,2}/\\d{4}$", "MM/dd/yyyy");
		put("^\\d{4}/\\d{1,2}/\\d{1,2}$", "yyyy/MM/dd");
		put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}$", "dd MMM yyyy");
		put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}$", "dd MMMM yyyy");
		put("^\\d{12}$", "yyyyMMddHHmm");
		put("^\\d{8}\\s\\d{4}$", "yyyyMMdd HHmm");
		put("^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}$", "dd-MM-yyyy HH:mm");
		put("^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}$", "yyyy-MM-dd HH:mm");
		put("^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}$", "MM/dd/yyyy HH:mm");
		put("^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}$", "yyyy/MM/dd HH:mm");
		put("^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}$", "dd MMM yyyy HH:mm");
		put("^\\d{1,2}\\s[a-z]{4,}\\s\\d{4}\\s\\d{1,2}:\\d{2}$", "dd MMMM yyyy HH:mm");
		put("^\\d{14}$", "yyyyMMddHHmmss");
		put("^\\d{8}\\s\\d{6}$", "yyyyMMdd HHmmss");
	}};

	public static long toMillis(String dateString) {
		return toMillis(dateString, getDateFormat(dateString));
	}

	public static long toMillis(String dateString, String dateFormat) {
		try {
			return new SimpleDateFormat(dateFormat).parse(dateString).getTime();
		}
		catch(ParseException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static String dateFormat(String dateString, String outputFormat) {
		try {
			return dateFormat(dateString, getDateFormat(dateString), outputFormat);
		}
		catch(NullPointerException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public static String dateFormat(String dateString, String inputFormat, String outputFormat) {
		try {
			Date value = new SimpleDateFormat(inputFormat).parse(dateString);
			return new SimpleDateFormat(outputFormat).format(value);
		}
		catch(ParseException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static String dateFormat(long date, String outputFormat) {
		return new SimpleDateFormat(outputFormat).format(new Date(date));
	}

	public static String[] copyAsStringToArray(String[] input, Object value) {
		String[] output = new String[input.length];
		Arrays.fill(output, String.valueOf(value));
		return output;
	}

	private static String getDateFormat (String dateString) {
		return DATE_FORMATS.keySet().parallelStream().filter(e -> dateString.toLowerCase().matches(e)).findFirst()
			.map(DATE_FORMATS::get).orElseThrow(() -> new NullPointerException("Unknown date format."));
	}

	@SuppressWarnings("unused")
	private static int findDateCol (FrameBlock block) {
		int cols = block.getNumColumns();
		int[] match_counter = new int[cols];
		int dateCol = -1;

		for (int i = 0; i < cols; i++) {
			String[] values = (String[]) block.getColumnData(i);
			int matchCount = 0;
			for (int j = 0; j < values.length; j++) {
				//skip null/blank entries
				if (values[j] == null || values[j].trim().isEmpty() || values[j].toUpperCase().equals("NULL") ||
					values[j].equals("0")) continue;
				String tmp = values[j];
				//check if value matches any date pattern
				if(DATE_FORMATS.keySet().parallelStream().anyMatch(e -> tmp.toLowerCase().matches(e))) matchCount++;
			}
			match_counter[i] = matchCount;
		}

		int maxMatches = Integer.MIN_VALUE;
		//get column with most matches -> date column
		for (int i = 0; i < match_counter.length; i++) {
			if (match_counter[i] > maxMatches) {
				maxMatches = match_counter[i];
				dateCol = i;
			}
		}

		if (maxMatches <= 0 || dateCol < 0){
			//ERROR - no date column found
			System.out.println("No date column in the dataset");
		}
		return dateCol;
	}
	public static String isDateColumn (String values) {
		return DATE_FORMATS.keySet().parallelStream().anyMatch(e -> values.toLowerCase().matches(e))?"1":"0";
	}
	public static String[] getDominantDateFormat (String[] values) {
		String[] output = new String[values.length];
		Map<String, String> date_formats = DATE_FORMATS;

		Map<String, Integer> format_matches = new HashMap<>();
		for (Map.Entry<String,String> entry : date_formats.entrySet()) {
			format_matches.put(entry.getValue(), 0);
		}

		for (int i = 0; i < values.length; i++) {
			//skip null/blank entries
			if (values[i] == null || values[i].trim().isEmpty() || values[i].equalsIgnoreCase("NULL")
				|| values[i].equals("0")) continue;
			String tmp = values[i];
			System.out.println("tmp "+tmp);
			String dateFormat = getDateFormat(tmp);
			//find pattern which matches values[i] -> increase count for this pattern
			format_matches.put(dateFormat, format_matches.get(dateFormat) + 1);
		}
		//find format with the most occurences in values -> dominant format
		String dominantFormat = format_matches.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getKey();
		for (int i = 0; i< values.length; i++){
			//skip null/blank entries
			if (values[i] == null || values[i].trim().isEmpty() ||
				values[i].equalsIgnoreCase("NULL") || values[i].equals("0")) continue;
			String currentFormat = getDateFormat(values[i]);
			//Locale.US needs to be used as otherwise dateformat like "dd MMM yyyy HH:mm" are not parsable
			SimpleDateFormat curr = new SimpleDateFormat(currentFormat, Locale.US);
			try {
				Date date = curr.parse(values[i]); //parse date string
				if (!currentFormat.equals(dominantFormat)){
					curr.applyPattern(dominantFormat);
				}
				//FIME: unused newDate
				//String newDate = curr.format(date); //convert date to dominant date format
				output[i] =  curr.format(date); //convert back to datestring
			} catch (ParseException e) {
				throw new DMLRuntimeException(e);
			}
		}

		return output;
	}

	public static String addTimeToDate (String dateString, int amountToAdd, String timeformat)  {
		String currentFormat = getDateFormat(dateString);
		Date date = null;
		SimpleDateFormat curr = new SimpleDateFormat(currentFormat, Locale.US);
		try {
			date = curr.parse(dateString);
		}
		catch(Exception e) { e.getMessage();}

		Date newDate;
		switch (timeformat) {
			case "ms": //milliseconds
				newDate = DateUtils.addMilliseconds(date, amountToAdd);
				break;
			case "m": //minutes
				newDate = DateUtils.addMinutes(date, amountToAdd);
				break;
			case "H": //hours
				newDate = DateUtils.addHours(date, amountToAdd);
				break;
			case "d": //days
				newDate = DateUtils.addDays(date, amountToAdd);
				break;
			case "w": //weeks
				newDate = DateUtils.addWeeks(date, amountToAdd);
				break;
			case "M": //months
				newDate = DateUtils.addMonths(date, amountToAdd);
				break;
			case "y": //years
				newDate = DateUtils.addYears(date, amountToAdd);
				break;
			default: //seconds
				newDate = DateUtils.addSeconds(date, amountToAdd);
				break;
		}
		return curr.format(newDate);
	}

	public static String[] getTimestamp(String[] values)
	{
		String output[] = new String[values.length];
		for (int i = 0; i< values.length; i++){
			//skip null/blank entries
			if (values[i] == null || values[i].trim().isEmpty() || values[i].toUpperCase().equals("NULL")
				|| values[i].equals("0")) continue;
			String currentFormat = getDateFormat(values[i]);
			//Locale.US needs to be used as otherwise dateformat like "dd MMM yyyy HH:mm" are not parsable
			SimpleDateFormat curr = new SimpleDateFormat(currentFormat, Locale.US);
			curr.setTimeZone(TimeZone.getTimeZone("UTC"));
			try {
				Date date = curr.parse(values[i]); //parse date string
				output[i] = String.valueOf(date.getTime()); //get timestamp in milliseconds
			} catch (ParseException e) {
				throw new DMLRuntimeException(e);
			}
		}
		return output;
	}

	public static String[] getSplittedStringAsArray (String input) {
		//Frame f = new Frame();
		String[] string_array = input.split("'[ ]*,[ ]*'");
		return string_array;//.subList(0,2);
	}
	
	public static double jaccardSim(String x, String y) {
		Set<String> charsX = new LinkedHashSet<>(Arrays.asList(x.split("(?!^)")));
		Set<String> charsY = new LinkedHashSet<>(Arrays.asList(y.split("(?!^)")));
	
		final int sa = charsX.size();
		final int sb = charsY.size();
		charsX.retainAll(charsY);
		final int intersection = charsX.size();
		return 1d / (sa + sb - charsX.size()) * intersection;
	}
	
	public static String columnStringToCSVString(String input, String separator) {
		StringBuffer sb = new StringBuffer(input);
		StringBuilder outStringBuilder = new StringBuilder();
		String[] string_array;
		
		// remove leading and trailing brackets: []
		int startOfArray = sb.indexOf("\"[");
		if (startOfArray >=0)
		  sb.delete(startOfArray, startOfArray + 2);
		
		
		int endOfArray = sb.lastIndexOf("]\"");
		if (endOfArray >=0) 
		  sb.delete(endOfArray, endOfArray + 2);
		
	
		// split values depending on their format
		if (sb.indexOf("'") != -1) { // string contains strings
		  // replace "None" with "'None'"
		  Pattern p = Pattern.compile(", None,");
		  Matcher m = p.matcher(sb);
		  string_array = m.replaceAll(", 'None',").split("'[ ]*,[ ]*'");
		
		  // remove apostrophe in first and last string element
		  string_array[0] = string_array[0].replaceFirst("'", "");
		  int lastArrayIndex = string_array.length - 1;
		  string_array[lastArrayIndex] = string_array[lastArrayIndex]
				.substring(0, string_array[lastArrayIndex].length() - 1);
		} 
		else  // string contains numbers only
		  string_array = sb.toString().split(",");
	
		// select a suitable separator that can be used to read in the file properly
		for(String s : string_array) 
			outStringBuilder.append(s).append(separator);
		
		outStringBuilder.delete(outStringBuilder.length() - separator.length(), outStringBuilder.length());
		return outStringBuilder.toString();
	}
	
	/**
	 * Generates a random FrameBlock with given parameters.
	 * 
	 * @param rows   frame rows
	 * @param cols   frame cols
	 * @param schema frame schema
	 * @param random random number generator
	 * @return FrameBlock
	 */
	public static FrameBlock generateRandomFrameBlock(int rows, int cols, ValueType[] schema, Random random) {
		String[] names = new String[cols];
		for(int i = 0; i < cols; i++)
			names[i] = schema[i].toString();
		FrameBlock frameBlock = new FrameBlock(schema, names);
		frameBlock.ensureAllocatedColumns(rows);
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				frameBlock.set(row, col, generateRandomValueFromValueType(schema[col], random));
		return frameBlock;
	}

	/**
	 * Generates a random value for a given Value Type
	 * 
	 * @param valueType the ValueType of which to generate the value
	 * @param random random number generator
	 * @return Object
	 */
	public static Object generateRandomValueFromValueType(ValueType valueType, Random random) {
		switch (valueType){
			case FP32:    return random.nextFloat();
			case FP64:    return random.nextDouble();
			case INT32:   return random.nextInt();
			case INT64:   return random.nextLong();
			case BOOLEAN: return random.nextBoolean();
			case STRING:
				return random.ints('a', 'z' + 1).limit(10)
					.collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
					.toString();
			default:
				return null;
		}
	}
	
	/**
	 * Generates a ValueType array from a String array
	 * 
	 * @param schemaValues the string schema of which to generate the ValueType
	 * @return ValueType[]
	 */
	public static ValueType[] stringToValueType(String[] schemaValues) {
		ValueType[] vt = new ValueType[schemaValues.length];
		for(int i=0; i < schemaValues.length; i++) {
			if(schemaValues[i].equalsIgnoreCase("STRING"))
				vt[i] = ValueType.STRING;
			else if (schemaValues[i].equalsIgnoreCase("FP64"))
				vt[i] = ValueType.FP64;
			else if (schemaValues[i].equalsIgnoreCase("FP32"))
				vt[i] = ValueType.FP32;
			else if (schemaValues[i].equalsIgnoreCase("INT64"))
				vt[i] = ValueType.INT64;
			else if (schemaValues[i].equalsIgnoreCase("INT32"))
				vt[i] = ValueType.INT32;
			else if (schemaValues[i].equalsIgnoreCase("BOOLEAN"))
				vt[i] = ValueType.BOOLEAN;
			else
				throw new DMLRuntimeException("Invalid column schema. Allowed values are STRING, FP64, FP32, INT64, INT32 and Boolean");
		}
		return vt;
	}

	public static int getEndIndex(int arrayLength, int startIndex, int blockSize){
		return blockSize <= 0 ? arrayLength : Math.min(arrayLength, startIndex + blockSize);
	}

	public static int[] getBlockSizes(int num, int numBlocks){
		int[] blockSizes = new int[numBlocks];
		Arrays.fill(blockSizes, num/numBlocks);
		for (int i = 0; i < num%numBlocks; i++){
			blockSizes[i]++;
		}
		return blockSizes;
	}

	public static String[] splitRecodeEntry(String s) {
		//forward to column encoder, as UtilFunctions available in map context
		return ColumnEncoderRecode.splitRecodeMapEntry(s);
	}

	public static String[] toStringArray(Object[] original) {
		String[] result = new String[original.length];
		for (int i = 0; i < result.length; i++)
			result[i] = String.valueOf(original[i]);
		return result;
	}
	
	public static <T> T getSafe(Future<T> task) {
		try {
			return task.get();
		} catch (InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static double[] convertStringToDoubleArray(String[] original) {
//		double[] ret = new double[original.length];
//		for (int i = 0; i < original.length; i++) {
//			try {
//				ret[i] = NumberFormat.getInstance().parse(original[i]).doubleValue();
//			}
//			catch(Exception e) {
//				e.printStackTrace();
//			}
//		}
//		return ret;

		return Arrays.stream(original).mapToDouble(Double::parseDouble).toArray();
	}
	
	/**
	 * Computes the word error rate (Levenshtein distance at word level):
	 * wer =  (numSubst + numDel + numIns) / length(r)
	 * 
	 * This code has been adapted from Apache Commons Lang 3.12 
	 * (getLevenshteinDistance, but for words instead of characters).
	 * 
	 * @param r reference string
	 * @param h hypothesis string
	 * @return word error rate (WER)
	 */
	public static double getWordErrorRate(String r, String h) {
		if (r == null || h == null) {
			throw new IllegalArgumentException("Strings must not be null");
		}

		//prepare string sequences 
		String[] s = r.split(" ");
		String[] t = h.split(" ");
		int n = s.length;
		int m = t.length;
		
		//basic size handling
		if( n == 0 || m == 0 )
			return Math.max(n, m);
		if (n > m) {
			// swap the input strings to consume less memory
			String[] tmp = s;
			s = t;
			t = tmp;
			n = m;
			m = t.length;
		}

		final int[] p = new int[n + 1];
		int i; // iterates through s
		int j; // iterates through t
		int upper_left;
		int upper;
		
		String t_j; // jth word of t
		int cost;
		for (i = 0; i <= n; i++) {
			p[i] = i;
		}
		for (j = 1; j <= m; j++) {
			upper_left = p[0];
			t_j = t[j - 1];
			p[0] = j;
			for (i = 1; i <= n; i++) {
				upper = p[i];
				cost = s[i - 1].equals(t_j) ? 0 : 1;
				p[i] = Math.min(Math.min(p[i - 1] + 1, p[i] + 1), upper_left + cost);
				upper_left = upper;
			}
		}
		//wer = number of edits / length
		return (double)p[n] / Math.max(n, m);
	}

	public static String[] cleanAndTokenizeRow(String[] row) {
		if (row == null || row.length == 0) {
			return new String[0];
		}
		StringBuilder sb = new StringBuilder();
		for (String s : row) {
			if (s != null) {
				sb.append(s).append(" ");
			}
		}
		String joined = sb.toString().trim().toLowerCase();  
		
		return joined.split("\\s+");
	}
	
	public static IndexedMatrixValue createIndexedMatrixBlock(MatrixBlock mb, DataCharacteristics mc, long ix) {
		try {
			//compute block indexes
			long blockRow = ix / mc.getNumColBlocks();
			long blockCol = ix % mc.getNumColBlocks();
			//compute block sizes
			int maxRow = UtilFunctions.computeBlockSize(mc.getRows(), blockRow+1, mc.getBlocksize());
			int maxCol = UtilFunctions.computeBlockSize(mc.getCols(), blockCol+1, mc.getBlocksize());
			//copy sub-matrix to block
			MatrixBlock block = new MatrixBlock(maxRow, maxCol, mb.isInSparseFormat());
			int row_offset = (int)blockRow*mc.getBlocksize();
			int col_offset = (int)blockCol*mc.getBlocksize();
			block = mb.slice( row_offset, row_offset+maxRow-1,
				col_offset, col_offset+maxCol-1, false, block );
			//create key-value pair
			return new IndexedMatrixValue(new MatrixIndexes(blockRow+1, blockCol+1), block);
		}
		catch(DMLRuntimeException ex) {
			throw new RuntimeException(ex);
		}
	}
}