/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark.utils;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.instructions.spark.data.SerLongWritable;
import org.tugraz.sysds.runtime.instructions.spark.data.SerText;
import org.tugraz.sysds.runtime.instructions.spark.functions.ConvertFrameBlockToIJVLines;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils.DataFrameExtractIDFunction;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesCSV;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.Pair;
import org.tugraz.sysds.runtime.matrix.mapred.FrameReblockBuffer;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.transform.TfUtils;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.FastStringTokenizer;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;



public class FrameRDDConverterUtils 
{
	private static final Log LOG = LogFactory.getLog(FrameRDDConverterUtils.class.getName());
	
	//=====================================
	// CSV <--> Binary block

	public static JavaPairRDD<Long, FrameBlock> csvToBinaryBlock(JavaSparkContext sc,
		JavaPairRDD<LongWritable, Text> input, DataCharacteristics mc, ValueType[] schema,
		boolean hasHeader, String delim, boolean fill, double fillValue)
	{
		//determine unknown dimensions and sparsity if required
		if( !mc.dimsKnown() ) { //nnz irrelevant here
 			JavaRDD<String> tmp = input.values()
					.map(new TextToStringFunction());
			String tmpStr = tmp.first();
			boolean metaHeader = tmpStr.startsWith(TfUtils.TXMTD_MVPREFIX) 
					|| tmpStr.startsWith(TfUtils.TXMTD_NDPREFIX);
			tmpStr = (metaHeader) ? tmpStr.substring(tmpStr.indexOf(delim)+1) : tmpStr;
			long rlen = tmp.count() - (hasHeader ? 1 : 0) - (metaHeader ? 2 : 0);
			long clen = IOUtilFunctions.splitCSV(tmpStr, delim).length;
			mc.set(rlen, clen, mc.getBlocksize(), -1);
		}
		
		//prepare csv w/ row indexes (sorted by filenames)
		JavaPairRDD<Text,Long> prepinput = input.values()
				.zipWithIndex(); //zip row index
		
		//prepare default schema if needed
		if( schema == null || schema.length==1 )
			schema = UtilFunctions.nCopies((int)mc.getCols(), ValueType.STRING);

		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<Long, FrameBlock> out = prepinput.mapPartitionsToPair(
				new CSVToBinaryBlockFunction(mc, schema, hasHeader, delim));
		
		return out;
	}

	public static JavaPairRDD<Long, FrameBlock> csvToBinaryBlock(JavaSparkContext sc,
		JavaRDD<String> input, DataCharacteristics mcOut, ValueType[] schema,
		boolean hasHeader, String delim, boolean fill, double fillValue)
	{
		//convert string rdd to serializable longwritable/text
		JavaPairRDD<LongWritable, Text> prepinput =
				input.mapToPair(new StringToSerTextFunction());
		
		//convert to binary block
		return csvToBinaryBlock(sc, prepinput, mcOut, schema, hasHeader, delim, fill, fillValue);
	}

	public static JavaRDD<String> binaryBlockToCsv(JavaPairRDD<Long,FrameBlock> in,
	                                               DataCharacteristics mcIn, FileFormatPropertiesCSV props, boolean strict)
	{
		JavaPairRDD<Long,FrameBlock> input = in;
		
		//sort if required (on blocks/rows)
		if( strict && !isSorted(input) ) {
			input = input.sortByKey(true);
		}
		
		//convert binary block to csv (from blocks/rows)
		return input.flatMap(
				new BinaryBlockToCSVFunction(props));
	}
	
	
	//=====================================
	// Text cell <--> Binary block

	public static JavaPairRDD<Long, FrameBlock> textCellToBinaryBlock(JavaSparkContext sc,
		JavaPairRDD<LongWritable, Text> in, DataCharacteristics mcOut, ValueType[] schema )
	{
		//convert input rdd to serializable long/frame block
		JavaPairRDD<Long,Text> input = 
				in.mapToPair(new LongWritableTextToLongTextFunction());
		//do actual conversion
		return textCellToBinaryBlockLongIndex(sc, input, mcOut, schema);
	}

	public static JavaPairRDD<Long, FrameBlock> textCellToBinaryBlockLongIndex(JavaSparkContext sc,
			JavaPairRDD<Long, Text> input, DataCharacteristics mc, ValueType[] schema )
	{
		//prepare default schema if needed
		if( schema == null || schema.length==1 ) {
			schema = UtilFunctions.nCopies((int)mc.getCols(), 
				(schema!=null) ? schema[0] : ValueType.STRING);
		}
		
 		//convert textcell rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<Long, FrameBlock> output = input.values()
				.mapPartitionsToPair(new TextToBinaryBlockFunction( mc, schema ));
		
		//aggregate partial matrix blocks
		return FrameRDDAggregateUtils.mergeByKey( output ); 
	}

	public static JavaRDD<String> binaryBlockToTextCell(JavaPairRDD<Long, FrameBlock> input, DataCharacteristics mcIn) {
		//convert frame blocks to ijv string triples  
		return input.flatMap(new ConvertFrameBlockToIJVLines());
	}
	
	//=====================================
	// Matrix block <--> Binary block

	public static JavaPairRDD<LongWritable, FrameBlock> matrixBlockToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixBlock> input, DataCharacteristics mcIn) {
		//convert and map to serializable LongWritable/frame block
		return matrixBlockToBinaryBlockLongIndex(sc,input, mcIn)
			.mapToPair(new LongFrameToLongWritableFrameFunction());
	}

	public static JavaPairRDD<Long, FrameBlock> matrixBlockToBinaryBlockLongIndex(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixBlock> input, DataCharacteristics dcIn) {
		JavaPairRDD<MatrixIndexes, MatrixBlock> in = input;
		DataCharacteristics mc = new MatrixCharacteristics(dcIn);
		
		//reblock matrix blocks if required (multiple column blocks)
		if(dcIn.getCols() > dcIn.getBlocksize()) {
			//split matrix blocks into extended matrix blocks 
			in = in.flatMapToPair(new MatrixFrameReblockFunction(dcIn));
			mc.setBlocksize(MatrixFrameReblockFunction.computeBlockSize(mc));
			
			//shuffle matrix blocks (instead of frame blocks) in order to exploit 
			//sparse formats (for sparse or wide matrices) during shuffle
			in = RDDAggregateUtils.mergeByKey(in, false);
		}
			
		//convert individual matrix blocks to frame blocks (w/o shuffle)
		return in.mapToPair(new MatrixToFrameBlockFunction(mc));
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockToMatrixBlock(JavaPairRDD<Long,FrameBlock> input,
		DataCharacteristics mcIn, DataCharacteristics mcOut)
	{
		//convert binary block to matrix block
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input
			.flatMapToPair(new BinaryBlockToMatrixBlockFunction(mcIn, mcOut));
	
		//aggregate partial matrix blocks
		return RDDAggregateUtils.mergeByKey(out, false); 
	}
	
	//=====================================
	// DataFrame <--> Binary block

	public static JavaPairRDD<Long, FrameBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			Dataset<Row> df, DataCharacteristics mc, boolean containsID) {
		return dataFrameToBinaryBlock(sc, df, mc, containsID, new Pair<String[], ValueType[]>());
	}

	public static JavaPairRDD<Long, FrameBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
		Dataset<Row> df, DataCharacteristics mc, boolean containsID, Pair<String[],ValueType[]> out)
	{
		//determine unknown dimensions if required
		if( !mc.dimsKnown() ) { //nnz are irrelevant here
			int colVect = getColVectFromDFSchema(df.schema(), containsID);
			int off = (containsID ? 1 : 0);
			long rlen = df.count();
			long clen = df.columns().length - off + ((colVect >= 0) ? 
					((Vector)df.first().get(off+colVect)).size() - 1 : 0);
			mc.set(rlen, clen, mc.getBlocksize(), -1);
		}
		
		//append or reuse row index column
		JavaPairRDD<Row, Long> prepinput = containsID ?
				df.javaRDD().mapToPair(new DataFrameExtractIDFunction(
					df.schema().fieldIndex(RDDConverterUtils.DF_ID_COLUMN))) :
				df.javaRDD().zipWithIndex(); //zip row index

		//convert data frame to frame schema (prepare once)
		String[] colnames = new String[(int)mc.getCols()];
		ValueType[] fschema = new ValueType[(int)mc.getCols()];
		int colVect = convertDFSchemaToFrameSchema(df.schema(), colnames, fschema, containsID);
		out.set(colnames, fschema); //make schema available
		
		//convert rdd to binary block rdd
		return prepinput.mapPartitionsToPair(
				new DataFrameToBinaryBlockFunction(mc, colnames, fschema, containsID, colVect));
	}

	public static Dataset<Row> binaryBlockToDataFrame(SparkSession sparkSession,
		JavaPairRDD<Long,FrameBlock> in, DataCharacteristics mc, ValueType[] schema)
	{
		if( !mc.colsKnown() )
			throw new RuntimeException("Number of columns needed to convert binary block to data frame.");
		
		//convert binary block to rows rdd 
		JavaRDD<Row> rowRDD = in.flatMap(
				new BinaryBlockToDataFrameFunction());
				
		//create data frame schema
		if( schema == null )
			schema = UtilFunctions.nCopies((int)mc.getCols(), ValueType.STRING);
		StructType dfSchema = convertFrameSchemaToDFSchema(schema, true);
	
		//rdd to data frame conversion
		return sparkSession.createDataFrame(rowRDD, dfSchema);
	}
	
	@Deprecated
	public static Dataset<Row> binaryBlockToDataFrame(SQLContext sqlContext,
		JavaPairRDD<Long,FrameBlock> in, DataCharacteristics mc, ValueType[] schema)
	{
		SparkSession sparkSession = sqlContext.sparkSession();
		return binaryBlockToDataFrame(sparkSession, in, mc, schema);
	}
	
	
	/**
	 * This function will convert Frame schema into DataFrame schema 
	 * 
	 * @param fschema frame schema
	 * @param containsID true if contains ID column
	 * @return Spark StructType of StructFields representing schema
	 */
	public static StructType convertFrameSchemaToDFSchema(ValueType[] fschema, boolean containsID)
	{
		// generate the schema based on the string of schema
		List<StructField> fields = new ArrayList<>();
		
		// add id column type
		if( containsID )
			fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, 
					DataTypes.DoubleType, true));
		
		// add remaining types
		int col = 1;
		for (ValueType schema : fschema) {
			DataType dt = null;
			switch(schema) {
				case STRING:  dt = DataTypes.StringType; break;
				case FP64:  dt = DataTypes.DoubleType; break;
				case INT64:     dt = DataTypes.LongType; break;
				case BOOLEAN: dt = DataTypes.BooleanType; break;
				default:      dt = DataTypes.StringType;
					LOG.warn("Using default type String for " + schema.toString());
			}
			fields.add(DataTypes.createStructField("C"+col++, dt, true));
		}
		
		return DataTypes.createStructType(fields);
	}
	
	/**
	 * NOTE: regarding the support of vector columns, we make the following 
	 * schema restriction: single vector column, which allows inference of
	 * the vector length without data access and covers the common case. 
	 * 
	 * @param dfschema schema as StructType
	 * @param colnames column names
	 * @param fschema array of SystemDS ValueTypes
	 * @param containsID if true, contains ID column
	 * @return 0-based column index of vector column, -1 if no vector.
	 */
	public static int convertDFSchemaToFrameSchema(StructType dfschema, String[] colnames, 
			ValueType[] fschema, boolean containsID)
	{
		//basic meta data
		int off = containsID ? 1 : 0;
		boolean containsVect = false;
		int lenVect = fschema.length - (dfschema.fields().length - off) + 1;
		int colVect = -1;
		
		//process individual columns
		for( int i=off, pos=0; i<dfschema.fields().length; i++ ) {
			StructField structType = dfschema.apply(i);
			colnames[pos] = structType.name();
			if(structType.dataType() == DataTypes.DoubleType 
				|| structType.dataType() == DataTypes.FloatType)
				fschema[pos++] = ValueType.FP64;
			else if(structType.dataType() == DataTypes.LongType 
				|| structType.dataType() == DataTypes.IntegerType)
				fschema[pos++] = ValueType.INT64;
			else if(structType.dataType() == DataTypes.BooleanType)
				fschema[pos++] = ValueType.BOOLEAN;
			else if(structType.dataType() instanceof VectorUDT) {
				if( containsVect )
					throw new RuntimeException("Found invalid second vector column.");
				String name = colnames[pos];
				colVect = pos;
				for( int j=0; j<lenVect; j++ ) {
					colnames[pos] = name+"v"+j;
					fschema[pos++] = ValueType.FP64;
				}
				containsVect = true;
			}
			else
				fschema[pos++] = ValueType.STRING;
		}
		
		return colVect;
	}
	
	/**
	 * Obtain column vector from DataFrame schema
	 * 
	 * @param dfschema schema as StructType
	 * @param containsID if true, contains ID column
	 * @return 0-based column index of vector column, -1 if no vector.
	 */
	private static int getColVectFromDFSchema(StructType dfschema, boolean containsID) {
		int off = containsID ? 1 : 0;
		for( int i=off; i<dfschema.fields().length; i++ ) {
			StructField structType = dfschema.apply(i);
			if(structType.dataType() instanceof VectorUDT)
				return i-off;
		}
		
		return -1;
	}
	
	/* 
	 * It will return JavaRDD<Row> based on csv data input file.
	 */
	public static JavaRDD<Row> csvToRowRDD(JavaSparkContext sc, String fnameIn, String delim, ValueType[] schema)
	{
		// Load a text file and convert each line to a java rdd.
		JavaRDD<String> dataRdd = sc.textFile(fnameIn);
		return dataRdd.map(new RowGenerator(schema, delim));
	}
	
	/* 
	 * It will return JavaRDD<Row> based on csv data.
	 */
	public static JavaRDD<Row> csvToRowRDD(JavaSparkContext sc, JavaRDD<String> dataRdd, String delim, ValueType[] schema)
	{
		// Convert each line to a java rdd.
		return dataRdd.map(new RowGenerator(schema, delim));
	}

	/* 
	 * Row Generator class based on individual line in CSV file.
	 */
	private static class RowGenerator implements Function<String,Row> 
	{
		private static final long serialVersionUID = -6736256507697511070L;

		private ValueType[] _schema = null;
		private String _delim = null; 
		
		public RowGenerator(ValueType[] schema, String delim) {
			_schema = schema;
			_delim = delim;
		}		
		 
		@Override
		public Row call(String record) throws Exception {
		      String[] fields = IOUtilFunctions.splitCSV(record, _delim);
		      Object[] objects = new Object[fields.length]; 
		      for (int i=0; i<fields.length; i++) {
			      objects[i] = UtilFunctions.stringToObject(_schema[i], fields[i]);
		      }
		      return RowFactory.create(objects);
		}
	}

	/**
	 * Check if the rdd is already sorted in order to avoid unnecessary
	 * sampling, shuffle, and sort per partition.
	 * 
	 * @param in JavaPairRDD of FrameBlocks
	 * @return true if RDD is sorted
	 */
	private static boolean isSorted(JavaPairRDD<Long, FrameBlock> in) {		
		//check sorted partitions (returns max key if true; -1 otherwise)
		List<Long> keys = in.keys().mapPartitions(
				new SortingAnalysisFunction()).collect();
		long max = 0;
		for( Long val : keys ) {
			if( val < max )
				return false;
			max = val;
		}
		return true;
	}

	private static class SortingAnalysisFunction implements FlatMapFunction<Iterator<Long>,Long> 
	{
		private static final long serialVersionUID = -5789003262381127469L;

		@Override
		public Iterator<Long> call(Iterator<Long> arg0) throws Exception 
		{
			long max = 0;
			while( max >= 0 && arg0.hasNext() ) {
				long val = arg0.next();
				max = (val < max) ? -1 : val;
			}
			ArrayList<Long> ret = new ArrayList<>();
			ret.add(max);
			return ret.iterator();
		}
	}
	
	
	/////////////////////////////////
	// CSV-SPECIFIC FUNCTIONS

	private static class StringToSerTextFunction implements PairFunction<String, LongWritable, Text> 
	{
		private static final long serialVersionUID = 8683232211035837695L;

		@Override
		public Tuple2<LongWritable, Text> call(String arg0) throws Exception {
			return new Tuple2<>(new SerLongWritable(1L), new SerText(arg0));
		}
	}

	public static class LongWritableToSerFunction implements PairFunction<Tuple2<LongWritable,FrameBlock>,LongWritable,FrameBlock> 
	{
		private static final long serialVersionUID = 2286037080400222528L;
		
		@Override
		public Tuple2<LongWritable, FrameBlock> call(Tuple2<LongWritable, FrameBlock> arg0) throws Exception  {
			return new Tuple2<>(new SerLongWritable(arg0._1.get()), arg0._2);
		}
	}

	public static class LongWritableTextToLongTextFunction implements PairFunction<Tuple2<LongWritable,Text>,Long,Text> 
	{
		private static final long serialVersionUID = -5408386071466175348L;

		@Override
		public Tuple2<Long, Text> call(Tuple2<LongWritable, Text> arg0) throws Exception  {
			return new Tuple2<>(new Long(arg0._1.get()), arg0._2);
		}
	}

	public static class LongFrameToLongWritableFrameFunction implements PairFunction<Tuple2<Long,FrameBlock>,LongWritable,FrameBlock> 
	{
		private static final long serialVersionUID = -1467314923206783333L;

		@Override
		public Tuple2<LongWritable, FrameBlock> call(Tuple2<Long, FrameBlock> arg0) throws Exception  {
			return new Tuple2<>(new LongWritable(arg0._1), arg0._2);
		}
	}

	public static class LongWritableFrameToLongFrameFunction implements PairFunction<Tuple2<LongWritable,FrameBlock>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = -1232439643533739078L;

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<LongWritable, FrameBlock> arg0) throws Exception  {
			return new Tuple2<>(arg0._1.get(), arg0._2);
		}
	}

	private static class TextToStringFunction implements Function<Text,String> 
	{
		private static final long serialVersionUID = -2744814934501782747L;

		@Override
		public String call(Text v1) throws Exception {
			return v1.toString();
		}
	}

	/**
	 * This functions allows to map rdd partitions of csv rows into a set of partial binary blocks.
	 * 
	 * NOTE: For this csv to binary block function, we need to hold all output blocks per partition 
	 * in-memory. Hence, we keep state of all column blocks and aggregate row segments into these blocks. 
	 * In terms of memory consumption this is better than creating partial blocks of row segments.
	 * 
	 */
	private static class CSVToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Text,Long>>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = -1976803898174960086L;

		private long _clen = -1;
		private boolean _hasHeader = false;
		private String _delim = null;
		private int _maxRowsPerBlock = -1; 
		private ValueType[] _schema = null;
		private String[] _colnames = null;
		private List<String> _mvMeta = null; //missing value meta data
		private List<String> _ndMeta = null; //num distinct meta data
		
		public CSVToBinaryBlockFunction(DataCharacteristics mc, ValueType[] schema, boolean hasHeader, String delim) {
			_clen = mc.getCols();
			_schema = schema;
			_hasHeader = hasHeader;
			_delim = delim;
			_maxRowsPerBlock = Math.max((int) (FrameBlock.BUFFER_SIZE/_clen), 1);
		}

		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Text,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<>();

			long ix = -1;
			FrameBlock fb = null;
			String[] tmprow = new String[(int)_clen]; 
			
			while( arg0.hasNext() )
			{
				Tuple2<Text,Long> tmp = arg0.next();
				String row = tmp._1().toString().trim();
				long rowix = tmp._2();
				if(_hasHeader && rowix == 0) { //Skip header
					_colnames = row.split(_delim);
					continue;
				}
				if( row.startsWith(TfUtils.TXMTD_MVPREFIX) ) {
					_mvMeta = Arrays.asList(Arrays.copyOfRange(IOUtilFunctions.splitCSV(row, _delim), 1, (int)_clen+1));
					continue;
				}
				else if( row.startsWith(TfUtils.TXMTD_NDPREFIX) ) {
					_ndMeta = Arrays.asList(Arrays.copyOfRange(IOUtilFunctions.splitCSV(row, _delim), 1, (int)_clen+1));
					continue;
				}
				
				//adjust row index for header and meta data
				rowix += (_hasHeader ? 0 : 1) - ((_mvMeta == null) ? 0 : 2);
				
				if( fb == null || fb.getNumRows() == _maxRowsPerBlock) {
					if( fb != null )
						flushBlocksToList(ix, fb, ret);
					ix = rowix;
					fb = createFrameBlock();
				}
				
				//split and process row data 
				fb.appendRow(IOUtilFunctions.splitCSV(row, _delim, tmprow));
			}
		
			//flush last blocks
			flushBlocksToList(ix, fb, ret);
		
			return ret.iterator();
		}
		
		// Creates new state of empty column blocks for current global row index.
		private FrameBlock createFrameBlock()
		{
			//frame block with given schema
			FrameBlock fb = new FrameBlock(_schema);
			
			//preallocate physical columns (to avoid re-allocations)
			fb.ensureAllocatedColumns(_maxRowsPerBlock);
			fb.reset(0, false); //reset data but keep schema
			fb.setNumRows(0);   //reset num rows to allow for append
			
			//handle meta data
			if( _colnames != null )
				fb.setColumnNames(_colnames);
			if( _mvMeta != null )
				for( int j=0; j<_clen; j++ )
					fb.getColumnMetadata(j).setMvValue(_mvMeta.get(j));
			if( _ndMeta != null )
				for( int j=0; j<_clen; j++ )
					fb.getColumnMetadata(j).setNumDistinct(Long.parseLong(_ndMeta.get(j)));
		
			return fb;
		}

		private static void flushBlocksToList( Long ix, FrameBlock fb, ArrayList<Tuple2<Long,FrameBlock>> ret ) {
			if( fb != null && fb.getNumRows()>=0 )
				ret.add(new Tuple2<>(ix, fb));
		}
	}

	private static class BinaryBlockToCSVFunction implements FlatMapFunction<Tuple2<Long,FrameBlock>,String> 
	{
		private static final long serialVersionUID = 8020608184930291069L;

		private FileFormatPropertiesCSV _props = null;
		
		public BinaryBlockToCSVFunction(FileFormatPropertiesCSV props) {
			_props = props;
		}

		@Override
		public Iterator<String> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			Long ix = arg0._1();
			FrameBlock blk = arg0._2();
			
			ArrayList<String> ret = new ArrayList<>();
			StringBuilder sb = new StringBuilder();
			
			//handle header information and frame meta data
			if( ix==1 ) {
				if( _props.hasHeader() ) {
					for(int j = 1; j <= blk.getNumColumns(); j++) {
						sb.append(blk.getColumnNames()[j] 
							+ ((j<blk.getNumColumns()-1)?_props.getDelim():""));
					}
					ret.add(sb.toString());
					sb.setLength(0); //reset
				}
				if( !blk.isColumnMetadataDefault() ) {
					sb.append(TfUtils.TXMTD_MVPREFIX + _props.getDelim());
					for( int j=0; j<blk.getNumColumns(); j++ )
						sb.append(blk.getColumnMetadata(j).getMvValue() + ((j<blk.getNumColumns()-1)?_props.getDelim():""));
					ret.add(sb.toString());
					sb.setLength(0); //reset
					sb.append(TfUtils.TXMTD_NDPREFIX + _props.getDelim());
					for( int j=0; j<blk.getNumColumns(); j++ )
						sb.append(blk.getColumnMetadata(j).getNumDistinct() + ((j<blk.getNumColumns()-1)?_props.getDelim():""));
					ret.add(sb.toString());
					sb.setLength(0); //reset		
				}
			}
		
			//handle Frame block data
			Iterator<String[]> iter = blk.getStringRowIterator();
			while( iter.hasNext() ) {
				String[] row = iter.next();
				for(int j=0; j<row.length; j++) {
					if(j != 0)
						sb.append(_props.getDelim());
					if(row[j] != null)
						sb.append(row[j]);
				}
				ret.add(sb.toString());
				sb.setLength(0); //reset
			}
			
			return ret.iterator();
		}
	}
	
	/////////////////////////////////
	// DataFrame-SPECIFIC FUNCTIONS
	
	private static class DataFrameToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Row,Long>>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = 2269315691094111843L;

		private long _clen = -1;
		private String[] _colnames = null;
		private ValueType[] _schema = null;
		private boolean _containsID = false;
		private int _colVect = -1;
		private int _maxRowsPerBlock = -1;
		
		public DataFrameToBinaryBlockFunction(DataCharacteristics mc, String[] colnames,
		                                      ValueType[] schema, boolean containsID, int colVect) {
			_clen = mc.getCols();
			_colnames = colnames;
			_schema = schema;
			_containsID = containsID;
			_colVect = colVect;
			_maxRowsPerBlock = Math.max((int) (FrameBlock.BUFFER_SIZE/_clen), 1);
		}
		
		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Row, Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<>();

			long ix = -1;
			FrameBlock fb = null;
			Object[] tmprow = new Object[(int)_clen];
			
			while( arg0.hasNext() )
			{
				Tuple2<Row,Long> tmp = arg0.next();
				Row row = tmp._1();
				long rowix = tmp._2()+1;
				
				if( fb == null || fb.getNumRows() == _maxRowsPerBlock) {
					if( fb != null )
						flushBlocksToList(ix, fb, ret);
					ix = rowix;
					fb = new FrameBlock(_schema, _colnames);
				}
				
				//process row data
				int off = _containsID ? 1 : 0;
				for(int i=off, pos=0; i<row.size(); i++) {
					if( i-off == _colVect ) {
						Vector vect = (Vector) row.get(i);
						for( int j=0; j<vect.size(); j++ )
							tmprow[pos++] = vect.apply(j);
					}
					else {
						tmprow[pos] = UtilFunctions.objectToObject(
							_schema[pos], row.get(i));
						pos++;
					}
				}
				fb.appendRow(tmprow);
			}
		
			//flush last blocks
			flushBlocksToList(ix, fb, ret);
		
			return ret.iterator();
		}

		private static void flushBlocksToList( Long ix, FrameBlock fb, ArrayList<Tuple2<Long,FrameBlock>> ret ) {
			if( fb != null && fb.getNumRows()>=0 )
				ret.add(new Tuple2<>(ix, fb));
		}
	}

	private static class BinaryBlockToDataFrameFunction implements FlatMapFunction<Tuple2<Long,FrameBlock>,Row> 
	{
		private static final long serialVersionUID = 8093340778966667460L;
		
		@Override
		public Iterator<Row> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			long rowIndex = arg0._1();
			FrameBlock blk = arg0._2();
			ArrayList<Row> ret = new ArrayList<>();

			//handle Frame block data
			int rows = blk.getNumRows();
			int cols = blk.getNumColumns();
			for( int i=0; i<rows; i++ ) {
				Object[] row = new Object[cols+1];
				row[0] = (double)rowIndex++;
				for( int j=0; j<cols; j++ )
					row[j+1] = blk.get(i, j);
				ret.add(RowFactory.create(row));
			}
			
			return ret.iterator();
		}
	}
	
	/////////////////////////////////
	// TEXTCELL-SPECIFIC FUNCTIONS
	
	private static abstract class CellToBinaryBlockFunction implements Serializable
	{
		private static final long serialVersionUID = -729614449626680946L;

		protected int _bufflen = -1;
		protected long _rlen = -1;
		protected long _clen = -1;
		
		protected CellToBinaryBlockFunction(DataCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, FrameBlock.BUFFER_SIZE);
		}

		protected void flushBufferToList( FrameReblockBuffer rbuff,  ArrayList<Tuple2<Long,FrameBlock>> ret ) 
			throws IOException, DMLRuntimeException
		{
			//temporary list of indexed matrix values to prevent library dependencies
			ArrayList<Pair<Long, FrameBlock>> rettmp = new ArrayList<>();
			rbuff.flushBufferToBinaryBlocks(rettmp);
			ret.addAll(SparkUtils.fromIndexedFrameBlock(rettmp));
		}
	}

	private static class TextToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Text>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = -2042208027876880588L;
		ValueType[] _schema = null;
		
		protected TextToBinaryBlockFunction(DataCharacteristics mc, ValueType[] schema ) {
			super(mc);
			_schema = schema;
		}

		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Iterator<Text> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<>();
			FrameReblockBuffer rbuff = new FrameReblockBuffer(_bufflen, _rlen, _clen, _schema );
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			while( arg0.hasNext() )
			{
				//get input string (ignore matrix market comments)
				String strVal = arg0.next().toString();
				if( strVal.startsWith("%") ) 
					continue;
				
				//parse input ijv triple
				st.reset( strVal );
				long row = st.nextLong();
				long col = st.nextLong();
				Object val = UtilFunctions.stringToObject(_schema[(int)col-1], st.nextToken());
				
				//flush buffer if necessary
				if( rbuff.getSize() >= rbuff.getCapacity() )
					flushBufferToList(rbuff, ret);
				
				//add value to reblock buffer
				rbuff.appendCell(row, col, val);
			}
			
			//final flush buffer
			flushBufferToList(rbuff, ret);
		
			return ret.iterator();
		}
	}
	
	// MATRIX Block <---> Binary Block specific functions
	private static class MatrixFrameReblockFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 6205071301074768437L;

		private int _blen = -1;
		private long _clen = -1;
		private int _maxRowsPerBlock = -1;
		private boolean _sparse = false;
		
		public MatrixFrameReblockFunction(DataCharacteristics dc) {
			_blen = dc.getBlocksize();
			_blen = dc.getBlocksize();
			_clen = dc.getCols();
			_maxRowsPerBlock = computeBlockSize(dc);
			_sparse = dc.dimsKnown() && MatrixBlock.evalSparseFormatInMemory(
					dc.getRows(), dc.getCols(), dc.getNonZeros()/(_clen/_blen));
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			MatrixBlock mbreuse = new MatrixBlock();
			boolean sparse = _sparse || mb.isInSparseFormat();
			
			//frame index (row id, 1-based)
			long rowix = (ix.getRowIndex()-1)*_blen+1;

			//global index within frame block (0-based)
			long cl = (int)((ix.getColumnIndex()-1)*_blen);
			long cu = Math.min(cl+mb.getNumColumns()-1, _clen);

			//prepare output frame blocks 
			for( int i=0; i<mb.getNumRows(); i+=_maxRowsPerBlock ) {
				int ru = Math.min(i+_maxRowsPerBlock, mb.getNumRows())-1;
				long rix = UtilFunctions.computeBlockIndex(rowix+i, _maxRowsPerBlock);
				MatrixIndexes ixout = new MatrixIndexes(rix, 1);
				MatrixBlock out = new MatrixBlock(ru-i+1, (int)_clen, sparse);
				out.copy(0, out.getNumRows()-1, (int)cl, (int)cu, 
					mb.slice(i, ru, 0, mb.getNumColumns()-1, mbreuse), true);
				out.examSparsity();
				ret.add(new Tuple2<>(ixout,out));
			}

			return ret.iterator();
		}
		
		/**
		 * Computes an output row block size that ensures matrix block alignment 
		 * and contains at most FrameBlock.BUFFER_SIZE cells per block.
		 * 
		 * @param dc matrix characteristics
		 * @return block size
		 */
		public static int computeBlockSize(DataCharacteristics dc) {
			int blen = dc.getBlocksize();
			int basic = Math.max((int)(FrameBlock.BUFFER_SIZE/dc.getCols()), 1);
			int div = (int)Math.ceil((double)blen/basic);
			while( blen % div != 0 ) 
				div++;
			return blen / div;
		}
	}

	private static class MatrixToFrameBlockFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,Long,FrameBlock>
	{
		private static final long serialVersionUID = 3716019666116660815L;

		private int _blen = -1;
			
		public MatrixToFrameBlockFunction(DataCharacteristics mc) {
			_blen = mc.getBlocksize();
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			FrameBlock fb = DataConverter.convertToFrameBlock(arg0._2());
			return  new Tuple2<>(
				(arg0._1().getRowIndex()-1)*_blen+1, fb);
		}
	}

	private static class BinaryBlockToMatrixBlockFunction implements PairFlatMapFunction<Tuple2<Long,FrameBlock>,MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -2654986510471835933L;
		
		private DataCharacteristics _mcIn;
		private DataCharacteristics _mcOut;

		public BinaryBlockToMatrixBlockFunction(DataCharacteristics mcIn, DataCharacteristics mcOut) {
			_mcIn = mcIn;		//Frame Characteristics
			_mcOut = mcOut;		//Matrix Characteristics
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			long rowIndex = arg0._1();
			FrameBlock blk = arg0._2();
			
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<>();
			long rlen = _mcIn.getRows();
			long clen = _mcIn.getCols();
			int blen = _mcOut.getBlocksize();
			
			//slice aligned matrix blocks out of given frame block
			long rstartix = UtilFunctions.computeBlockIndex(rowIndex, blen);
			long rendix = UtilFunctions.computeBlockIndex(rowIndex+blk.getNumRows()-1, blen);
			long cendix = UtilFunctions.computeBlockIndex(blk.getNumColumns(), blen);
			for( long rix=rstartix; rix<=rendix; rix++ ) { //for all row blocks
				long rpos = UtilFunctions.computeCellIndex(rix, blen, 0);
				int lrlen = UtilFunctions.computeBlockSize(rlen, rix, blen);
				int fix = (int)((rpos-rowIndex>=0) ? rpos-rowIndex : 0);
				int fix2 = (int)Math.min(rpos+lrlen-rowIndex-1,blk.getNumRows()-1);
				int mix = UtilFunctions.computeCellInBlock(rowIndex+fix, blen);
				int mix2 = mix + (fix2-fix);
				for( long cix=1; cix<=cendix; cix++ ) { //for all column blocks
					long cpos = UtilFunctions.computeCellIndex(cix, blen, 0);
					int lclen = UtilFunctions.computeBlockSize(clen, cix, blen);
					MatrixBlock matrix = new MatrixBlock(lrlen, lclen, false);
					FrameBlock frame = blk.slice(fix, fix2, 
							(int)cpos-1, (int)cpos+lclen-2, new FrameBlock());
					MatrixBlock mframe = DataConverter.convertToMatrixBlock(frame);
					ret.add(new Tuple2<>(new MatrixIndexes(rix, cix), 
							matrix.leftIndexingOperations(mframe, mix, mix2, 0, lclen-1, 
							new MatrixBlock(), UpdateType.INPLACE_PINNED)));
				}
			}

			return ret.iterator();
		}
	}
	
}
