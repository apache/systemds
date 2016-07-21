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

package org.apache.sysml.runtime.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.wink.json4j.JSONArray;

public class UtilFunctions 
{
	//for accurate cast of double values to int and long 
	//IEEE754: binary64 (double precision) eps = 2^(-53) = 1.11 * 10^(-16)
	//(same epsilon as used for matrix index cast in R)
	public static double DOUBLE_EPS = Math.pow(2, -53);
	
	
	public static int longHashFunc(long v) {
		return (int)(v^(v>>>32));
	}
	
	public static int nextIntPow2( int in ) {
		int expon = (in==0) ? 0 : 32-Integer.numberOfLeadingZeros(in-1);
		long pow2 = (long) Math.pow(2, expon);
		return (int)((pow2>Integer.MAX_VALUE)?Integer.MAX_VALUE : pow2);	
	}
	
	/**
	 * Computes the 1-based block index based on the global cell index and block size meta
	 * data. See computeCellIndex for the inverse operation.
	 * 
	 * @param cellIndex
	 * @param blockSize
	 * @return
	 */
	public static long computeBlockIndex(long cellIndex, int blockSize) {
		return (cellIndex-1)/blockSize + 1;
	}
	
	/**
	 * Computes the 0-based cell-in-block index based on the global cell index and block
	 * size meta data. See computeCellIndex for the inverse operation.
	 * 
	 * @param cellIndex
	 * @param blockSize
	 * @return
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
	 * @param blockIndex
	 * @param blockSize
	 * @param cellInBlock
	 * @return
	 */
	public static long computeCellIndex( long blockIndex, int blockSize, int cellInBlock ) {
		return (blockIndex-1)*blockSize + 1 + cellInBlock;
	}
	
	/**
	 * Computes the actual block size based on matrix dimension, block index, and block size
	 * meta data. For boundary blocks, the actual block size is less or equal than the block 
	 * size meta data; otherwise they are identical.  
	 *  
	 * @param len
	 * @param blockIndex
	 * @param blockSize
	 * @return
	 */
	public static int computeBlockSize( long len, long blockIndex, long blockSize ) {
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
	
	/**
	 * 
	 * @param ix
	 * @param brlen
	 * @param bclen
	 * @param rl
	 * @param ru
	 * @return
	 */
	public static boolean isInFrameBlockRange( Long ix, int brlen, long rl, long ru )
	{
		if(rl > ix+brlen-1 || ru < ix)
			return false;
		else
			return true;
	}
	
	/**
	 * 
	 * @param ix
	 * @param brlen
	 * @param bclen
	 * @param ixrange
	 * @return
	 */
	public static boolean isInBlockRange( MatrixIndexes ix, int brlen, int bclen, IndexRange ixrange )
	{
		return isInBlockRange(ix, brlen, bclen, 
				ixrange.rowStart, ixrange.rowEnd, 
				ixrange.colStart, ixrange.colEnd);
	}
	
	/**
	 * 
	 * @param ix
	 * @param brlen
	 * @param bclen
	 * @param ixrange
	 * @return
	 */
	public static boolean isInFrameBlockRange( Long ix, int brlen, int bclen, IndexRange ixrange )
	{
		return isInFrameBlockRange(ix, brlen, ixrange.rowStart, ixrange.rowEnd);
	}
	
	// Reused by both MR and Spark for performing zero out
	public static IndexRange getSelectedRangeForZeroOut(IndexedMatrixValue in, int blockRowFactor, int blockColFactor, IndexRange indexRange) 
	{
		IndexRange tempRange = new IndexRange(-1, -1, -1, -1);
		
		long topBlockRowIndex=UtilFunctions.computeBlockIndex(indexRange.rowStart, blockRowFactor);
		int topRowInTopBlock=UtilFunctions.computeCellInBlock(indexRange.rowStart, blockRowFactor);
		long bottomBlockRowIndex=UtilFunctions.computeBlockIndex(indexRange.rowEnd, blockRowFactor);
		int bottomRowInBottomBlock=UtilFunctions.computeCellInBlock(indexRange.rowEnd, blockRowFactor);
		
		long leftBlockColIndex=UtilFunctions.computeBlockIndex(indexRange.colStart, blockColFactor);
		int leftColInLeftBlock=UtilFunctions.computeCellInBlock(indexRange.colStart, blockColFactor);
		long rightBlockColIndex=UtilFunctions.computeBlockIndex(indexRange.colEnd, blockColFactor);
		int rightColInRightBlock=UtilFunctions.computeCellInBlock(indexRange.colEnd, blockColFactor);
		
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
	public static IndexRange getSelectedRangeForZeroOut(Pair<Long, FrameBlock> in, int blockRowFactor, int blockColFactor, IndexRange indexRange, long lSrcRowIndex, long lDestRowIndex) 
	{
		int iRowStart, iRowEnd, iColStart, iColEnd;
		
		if(indexRange.rowStart <= lDestRowIndex)
			iRowStart = 0;
		else
			iRowStart = (int) (indexRange.rowStart - in.getKey());
		iRowEnd = (int) Math.min(indexRange.rowEnd - lSrcRowIndex, blockRowFactor)-1;
		
		iColStart = UtilFunctions.computeCellInBlock(indexRange.colStart, blockColFactor);
		iColEnd = UtilFunctions.computeCellInBlock(indexRange.colEnd, blockColFactor);

		return  new IndexRange(iRowStart, iRowEnd, iColStart, iColEnd);
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
	
	public static int toInt(Object obj)
	{
		if( obj instanceof Long )
			return ((Long)obj).intValue();
		else
			return ((Integer)obj).intValue();
	}
	
	public static int roundToNext(int val, int factor) {
		//round up to next non-zero multiple of factor
		int pval = Math.max(val, factor);
		return ((pval + factor-1) / factor) * factor;
	}

	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	public static Object doubleToObject(ValueType vt, double in) {
		return doubleToObject(vt, in, true);
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @param sparse
	 * @return
	 */
	public static Object doubleToObject(ValueType vt, double in, boolean sparse) {
		if( in == 0 && sparse) return null;
		switch( vt ) {
			case STRING:  return String.valueOf(in);
			case BOOLEAN: return (in!=0);
			case INT:     return UtilFunctions.toLong(in);
			case DOUBLE:  return in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	public static Object stringToObject(ValueType vt, String in) {
		if( in == null )  return null;
		switch( vt ) {
			case STRING:  return in;
			case BOOLEAN: return Boolean.parseBoolean(in);
			case INT:     return Long.parseLong(in);
			case DOUBLE:  return Double.parseDouble(in);
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	public static double objectToDouble(ValueType vt, Object in) {
		if( in == null )  return 0;
		switch( vt ) {
			case STRING:  return !((String)in).isEmpty() ? Double.parseDouble((String)in) : 0;
			case BOOLEAN: return ((Boolean)in)?1d:0d;
			case INT:     return (Long)in;
			case DOUBLE:  return (Double)in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static String objectToString( Object in ) {
		return (in !=null) ? in.toString() : null;
	}
	
	/**
	 * 
	 * @param in
	 * @param ignoreNull	
	 * 		If this flag has set, it will ignore null. This flag is mainly used in merge functionality to override data with "null" data.
	 * @return
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
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	public static Object objectToObject(ValueType vt, Object in ) {
		String str = objectToString(in);
		return stringToObject(vt, str );
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	public static Object objectToObject(ValueType vt, Object in, boolean ignoreNull ) {
		String str = objectToString(in, ignoreNull);
		if (str==null || vt == ValueType.STRING)
			return str;
		else
			return stringToObject(vt, str); 
	}	
	
	/**
	 * 
	 * @param vt
	 * @param in1
	 * @param in2
	 * 
	 * @return
	 */
	public static int compareTo(ValueType vt, Object in1, Object in2) {
		if(in1 == null && in2 == null) return 0;
		else if(in1 == null) return -1;
		else if(in2 == null) return 1;
 
		switch( vt ) {
			case STRING:  return ((String)in1).compareTo((String)in2);
			case BOOLEAN: return ((Boolean)in1).compareTo((Boolean)in2);
			case INT:     return ((Long)in1).compareTo((Long)in2);
			case DOUBLE:  return ((Double)in1).compareTo((Double)in2);
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}

	/**
	 * Compares two version strings of format x.y.z, where x is major,
	 * y is minor, and z is maintenance release.
	 * 
	 * @param version1
	 * @param version2
	 * @return
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
				&& s.length() >=2 && ((s.startsWith("\"") && s.endsWith("\"")) 
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
	
	/**
	 * Parses a memory size with optional g/m/k quantifiers into its
	 * number representation.
	 * 
	 * @param arg
	 * @return
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
	 * 
	 * @param low   lower bound (inclusive)
	 * @param up    upper bound (inclusive)
	 * @param incr  increment 
	 * @return
	 */
	public static List<Integer> getSequenceList(int low, int up, int incr) {
		ArrayList<Integer> ret = new ArrayList<Integer>();
		for( int i=low; i<=up; i+=incr )
			ret.add(i);
		return ret;
	}

	/**
	 * Returns the schema based on Json object
	 * 
	 * @param schemaObject
	 * @return
	 */
	public static List<ValueType> getSchemaType(Object schemaObject)
	{
		JSONArray schemaJsonArr = (JSONArray)schemaObject;
		ValueType[] schemaArray = new ValueType[schemaJsonArr.size()];
		
		for(int i=0; i < schemaJsonArr.length(); i++)
				schemaArray[i] = ValueType.valueOf((String)schemaJsonArr.get(0));
		return Arrays.asList(schemaArray);
	}
	
	/**
	 * Returns the subset of the schema 
	 * 
	 * @param srcSchema
	 * @param lStart
	 * @param lEnd
	 * 
	 * @return
	 */
	public static List<ValueType> getSubSchema(List<ValueType> srcSchema, long lStart, long lEnd)
	{
		ValueType [] schema = new ValueType[(int) (lEnd-lStart+1)];
		for(int i = 0; i < schema.length; i++)
			schema[i] = srcSchema.get((int) (lStart+i));
		
		return Arrays.asList(schema);
	}
	
	
	/*
	 * This function will return datatype, if its Matrix or Frame
	 * 
	 *  @param	str
	 *  		Instruction string to execute
	 */
	
	public static DataType getDataType(String str, int index)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		CPOperand in1 = new CPOperand(parts[index]);
	
		return in1.getDataType();
	}
	
	/*
	 * This function will convert Frame schema into DataFrame schema 
	 * 
	 *  @param	schema
	 *  		Frame schema in the form of List<ValueType>
	 *  @return
	 *  		Returns the DataFrame schema (StructType)
	 */
	public static StructType convertFrameSchemaToDFSchema(List<ValueType> lschema)
	{
		// Generate the schema based on the string of schema
		List<StructField> fields = new ArrayList<StructField>();
		
		int i = 1;
		for (ValueType schema : lschema)
		{
			org.apache.spark.sql.types.DataType dataType = DataTypes.StringType;
			switch(schema)
			{
				case STRING:
					dataType = DataTypes.StringType;
					break;
				case DOUBLE:
					dataType = DataTypes.DoubleType;
					break;
				case INT:
					dataType = DataTypes.LongType;
					break;
				case BOOLEAN:
					dataType = DataTypes.BooleanType;
					break;
				default:
					System.out.println("Default schema type is String.");
			}
			fields.add(DataTypes.createStructField("C"+i++, dataType, true));
		}
		
		return DataTypes.createStructType(fields);
	}
	
}
