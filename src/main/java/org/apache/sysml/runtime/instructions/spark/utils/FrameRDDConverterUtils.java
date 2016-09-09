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

package org.apache.sysml.runtime.instructions.spark.utils;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Arrays;

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
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.instructions.spark.data.SerLongWritable;
import org.apache.sysml.runtime.instructions.spark.data.SerText;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertFrameBlockToIJVLines;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.FrameReblockBuffer;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.UtilFunctions;



public class FrameRDDConverterUtils 
{
	private static final Log LOG = LogFactory.getLog(FrameRDDConverterUtils.class.getName());
	
	//=====================================
	// CSV <--> Binary block

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> csvToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> input, MatrixCharacteristics mcOut, 
			boolean hasHeader, String delim, boolean fill, double fillValue) 
		throws DMLRuntimeException 
	{
		//determine unknown dimensions and sparsity if required
		if( !mcOut.dimsKnown(true) ) {
			JavaRDD<String> tmp = input.values()
					.map(new TextToStringFunction());
			String tmpStr = tmp.first();
			boolean metaHeader = tmpStr.startsWith(TfUtils.TXMTD_MVPREFIX) 
					|| tmpStr.startsWith(TfUtils.TXMTD_NDPREFIX);
			tmpStr = (metaHeader) ? tmpStr.substring(tmpStr.indexOf(delim)+1) : tmpStr;
			long rlen = tmp.count() - (hasHeader ? 1 : 0) - (metaHeader ? 2 : 0);
			long clen = IOUtilFunctions.splitCSV(tmpStr, delim).length;
			mcOut.set(rlen, clen, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), -1);
		}
		
		//prepare csv w/ row indexes (sorted by filenames)
		JavaPairRDD<Text,Long> prepinput = input.values()
				.zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<Long, FrameBlock> out = prepinput
				.mapPartitionsToPair(new CSVToBinaryBlockFunction(mcOut, hasHeader, delim, fill));
		
		return out;
	}
	
	/**
	 * @param sc 
	 * @param input
	 * @param mcOut
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param fillValue
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> csvToBinaryBlock(JavaSparkContext sc,
			JavaRDD<String> input, MatrixCharacteristics mcOut, 
			boolean hasHeader, String delim, boolean fill, double fillValue) 
		throws DMLRuntimeException 
	{
		//convert string rdd to serializable longwritable/text
		JavaPairRDD<LongWritable, Text> prepinput =
				input.mapToPair(new StringToSerTextFunction());
		
		//convert to binary block
		return csvToBinaryBlock(sc, prepinput, mcOut, hasHeader, delim, fill, fillValue);
	}
	
	/**
	 * 
	 * @param in
	 * @param mcIn
	 * @param props
	 * @param strict
	 * @return
	 */
	public static JavaRDD<String> binaryBlockToCsv(JavaPairRDD<Long,FrameBlock> in, MatrixCharacteristics mcIn, CSVFileFormatProperties props, boolean strict)
	{
		JavaPairRDD<Long,FrameBlock> input = in;
		
		//sort if required (on blocks/rows)
		if( strict ) {
			input = input.sortByKey(true);
		}
		
		//convert binary block to csv (from blocks/rows)
		JavaRDD<String> out = input
				.flatMap(new BinaryBlockToCSVFunction(props));
	
		return out;
	}
	
	
	//=====================================
	// Text cell <--> Binary block
	
	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param schema
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> textCellToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> in, MatrixCharacteristics mcOut, List<ValueType> schema ) 
		throws DMLRuntimeException  
	{
		//replicate schema entry if necessary
		List<ValueType> lschema = (schema.size()==1 && mcOut.getCols()>1) ?
				Collections.nCopies((int)mcOut.getCols(), schema.get(0)) : schema;
		
		//convert input rdd to serializable long/frame block
		JavaPairRDD<Long,Text> input = 
				in.mapToPair(new LongWritableTextToLongTextFunction());
		
		//do actual conversion
		return textCellToBinaryBlockLongIndex(sc, input, mcOut, lschema);
	}

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param schema
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> textCellToBinaryBlockLongIndex(JavaSparkContext sc,
			JavaPairRDD<Long, Text> input, MatrixCharacteristics mcOut, List<ValueType> schema ) 
		throws DMLRuntimeException  
	{
		
 		//convert textcell rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<Long, FrameBlock> output = input.values().mapPartitionsToPair(new TextToBinaryBlockFunction( mcOut, schema ));
		
		//aggregate partial matrix blocks
		JavaPairRDD<Long,FrameBlock> out = 
				(JavaPairRDD<Long, FrameBlock>) RDDAggregateUtils.mergeByFrameKey( output ); 

		return out;
	}

	/**
	 * 
	 * @param input
	 * @param mcIn
	 * @param format
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaRDD<String> binaryBlockToTextCell(JavaPairRDD<Long, FrameBlock> input, MatrixCharacteristics mcIn) 
		throws DMLRuntimeException 
	{
		//convert frame blocks to ijv string triples  
		return input.flatMap(new ConvertFrameBlockToIJVLines());
	}
	
	//=====================================
	// Matrix block <--> Binary block

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcIn
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<LongWritable, FrameBlock> matrixBlockToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixBlock> input, MatrixCharacteristics mcIn)
		throws DMLRuntimeException 
	{
		//Do actual conversion
		JavaPairRDD<Long, FrameBlock> output = matrixBlockToBinaryBlockLongIndex(sc,input, mcIn);
		
		//convert input rdd to serializable LongWritable/frame block
		JavaPairRDD<LongWritable,FrameBlock> out = 
				output.mapToPair(new LongFrameToLongWritableFrameFunction());
		
		return out;
	}
	

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcIn
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> matrixBlockToBinaryBlockLongIndex(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixBlock> input, MatrixCharacteristics mcIn)
		throws DMLRuntimeException 
	{
		JavaPairRDD<Long, FrameBlock> out = null;
		
		if(mcIn.getCols() > mcIn.getColsPerBlock()) {
			
			out = input.flatMapToPair(new MatrixToBinaryBlockFunction(mcIn));
			
			//aggregate partial frame blocks
			if(mcIn.getCols() > mcIn.getColsPerBlock())
				out = (JavaPairRDD<Long, FrameBlock>) RDDAggregateUtils.mergeByFrameKey( out );
		}
		else
			out = input.mapToPair(new MatrixToBinaryBlockOneColumnBlockFunction(mcIn));
		
		return out;
	}
	

	
	/**
	 * 
	 * @param in
	 * @param mcIn
	 * @param props
	 * @param strict
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockToMatrixBlock(JavaPairRDD<Long,FrameBlock> input, 
			MatrixCharacteristics mcIn, MatrixCharacteristics mcOut) 
	{
		//convert binary block to matrix block
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input
				.flatMapToPair(new BinaryBlockToMatrixBlockFunction(mcIn, mcOut));
	
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 	
		
		return out;
	}
	
	//=====================================
	// DataFrame <--> Binary block

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<Long, FrameBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID) 
		throws DMLRuntimeException 
	{
		
		if(containsID)
			df = df.drop("__INDEX");
		
		//determine unknown dimensions if required
		if( !mcOut.dimsKnown(true) ) {
			JavaRDD<Row> tmp = df.javaRDD();
			long rlen = tmp.count();
			long clen = containsID ? (df.columns().length - 1) : df.columns().length;
			mcOut.set(rlen, clen, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), -1);
		}
		
		JavaPairRDD<Row, Long> prepinput = df.javaRDD()
				.zipWithIndex(); //zip row index
		
		//convert rdd to binary block rdd
		JavaPairRDD<Long, FrameBlock> out = prepinput
				.mapPartitionsToPair(new DataFrameToBinaryBlockFunction(mcOut));
		
		return out;
	}

	/**
	 * 
	 * @param in
	 * @param mcIn
	 * @param props
	 * @param strict
	 * @return
	 */
	public static DataFrame binaryBlockToDataFrame(JavaPairRDD<Long,FrameBlock> in, MatrixCharacteristics mcIn, JavaSparkContext sc)
	{
		List<ValueType> schema = in.first()._2().getSchema();
		
		//convert binary block to rows rdd (from blocks/rows)
		JavaRDD<Row> rowRDD = in.flatMap(new BinaryBlockToDataFrameFunction());
				
		SQLContext sqlContext = new SQLContext(sc);
		StructType dfSchema = convertFrameSchemaToDFSchema(schema);
		DataFrame df = sqlContext.createDataFrame(rowRDD, dfSchema);
	
		return df;
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
		for (ValueType schema : lschema) {
			org.apache.spark.sql.types.DataType dataType = DataTypes.StringType;
			switch(schema) {
				case STRING:  dataType = DataTypes.StringType; break;
				case DOUBLE:  dataType = DataTypes.DoubleType; break;
				case INT:     dataType = DataTypes.LongType; break;
				case BOOLEAN: dataType = DataTypes.BooleanType; break;
				default:
					LOG.warn("Using default type String for " + schema.toString());
			}
			fields.add(DataTypes.createStructField("C"+i++, dataType, true));
		}
		
		return DataTypes.createStructType(fields);
	}
	
	/* 
	 * It will return JavaRDD<Row> based on csv data input file.
	 */
	public static JavaRDD<Row> getRowRDD(JavaSparkContext sc, String fnameIn, String delim, List<ValueType> schema)
	{
		// Load a text file and convert each line to a java rdd.
		JavaRDD<String> dataRdd = sc.textFile(fnameIn);
		return dataRdd.map(new RowGenerator(schema, delim));
	}
	
	/* 
	 * Row Generator class based on individual line in CSV file.
	 */
	private static class RowGenerator implements Function<String,Row> 
	{
		private static final long serialVersionUID = -6736256507697511070L;

		private List<ValueType> _schema = null;
		private String _delim = null; 
		
		public RowGenerator(List<ValueType> schema, String delim) {
			_schema = schema;
			_delim = delim;
		}		
		 
		@Override
		public Row call(String record) throws Exception {
		      String[] fields = IOUtilFunctions.splitCSV(record, _delim);
		      Object[] objects = new Object[fields.length]; 
		      for (int i=0; i<fields.length; i++) {
			      objects[i] = UtilFunctions.stringToObject(_schema.get(i), fields[i]);
		      }
		      return RowFactory.create(objects);
		}
	}
	

	/////////////////////////////////
	// CSV-SPECIFIC FUNCTIONS
	
	/**
	 * 
	 */
	private static class StringToSerTextFunction implements PairFunction<String, LongWritable, Text> 
	{
		private static final long serialVersionUID = 8683232211035837695L;

		@Override
		public Tuple2<LongWritable, Text> call(String arg0) throws Exception {
			return new Tuple2<LongWritable,Text>(new SerLongWritable(1L), new SerText(arg0));
		}
	}
	
	/**
	 * 
	 */
	public static class LongWritableToSerFunction implements PairFunction<Tuple2<LongWritable,FrameBlock>,LongWritable,FrameBlock> 
	{
		private static final long serialVersionUID = 2286037080400222528L;
		
		@Override
		public Tuple2<LongWritable, FrameBlock> call(Tuple2<LongWritable, FrameBlock> arg0) throws Exception  {
			return new Tuple2<LongWritable,FrameBlock>(new SerLongWritable(arg0._1.get()), arg0._2);
		}
	}
	
	/**
	 * 
	 */
	public static class LongWritableTextToLongTextFunction implements PairFunction<Tuple2<LongWritable,Text>,Long,Text> 
	{
		private static final long serialVersionUID = -5408386071466175348L;

		@Override
		public Tuple2<Long, Text> call(Tuple2<LongWritable, Text> arg0) throws Exception  {
			return new Tuple2<Long,Text>(new Long(arg0._1.get()), arg0._2);
		}
	}
	
	/**
	 * 
	 */
	public static class LongFrameToLongWritableFrameFunction implements PairFunction<Tuple2<Long,FrameBlock>,LongWritable,FrameBlock> 
	{
		private static final long serialVersionUID = -1467314923206783333L;

		@Override
		public Tuple2<LongWritable, FrameBlock> call(Tuple2<Long, FrameBlock> arg0) throws Exception  {
			return new Tuple2<LongWritable, FrameBlock>(new LongWritable(arg0._1), arg0._2);
		}
	}

	/**
	 * 
	 */
	public static class LongWritableFrameToLongFrameFunction implements PairFunction<Tuple2<LongWritable,FrameBlock>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = -1232439643533739078L;

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<LongWritable, FrameBlock> arg0) throws Exception  {
			return new Tuple2<Long, FrameBlock>(arg0._1.get(), arg0._2);
		}
	}
	
	/**
	 * 
	 */
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
		private boolean _fill = false;
		private int _maxRowsPerBlock = -1; 
		private List<String> _colnames = null;
		private List<String> _mvMeta = null; //missing value meta data
		private List<String> _ndMeta = null; //num distinct meta data
		
		public CSVToBinaryBlockFunction(MatrixCharacteristics mc, boolean hasHeader, String delim, boolean fill) {
			_clen = mc.getCols();
			_hasHeader = hasHeader;
			_delim = delim;
			_fill = fill;
			_maxRowsPerBlock = Math.max((int) (FrameBlock.BUFFER_SIZE/_clen), 1);
		}

		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Text,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<Tuple2<Long,FrameBlock>>();

			Long[] ix = new Long[1];
			FrameBlock[] mb = new FrameBlock[1];
			int iRowsInBlock = 0;
			
			while( arg0.hasNext() )
			{
				Tuple2<Text,Long> tmp = arg0.next();
				String row = tmp._1().toString().trim();
				long rowix = tmp._2();
				if(_hasHeader && rowix == 0) { //Skip header
					_colnames = Arrays.asList(IOUtilFunctions.splitCSV(row, _delim));
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
				
				if( iRowsInBlock == 0 || iRowsInBlock == _maxRowsPerBlock) {
					if( iRowsInBlock == _maxRowsPerBlock )
						flushBlocksToList(ix, mb, ret);
					createBlocks(rowix, ix, mb);
					iRowsInBlock = 0;
				}
				
				//process row data
				String[] parts = IOUtilFunctions.splitCSV(row, _delim);
				boolean emptyFound = false;
				mb[0].appendRow(parts);
				iRowsInBlock++;
		
				//sanity check empty cells filled w/ values
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(row, _fill, emptyFound);
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret;
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, Long[] ix, FrameBlock[] mb)
		{
			//compute row block index and number of column blocks
			ix[0] = rowix;
			mb[0] = new FrameBlock((int)_clen, ValueType.STRING);
			if( _colnames != null )
				mb[0].setColumnNames(_colnames);
			if( _mvMeta != null )
				for( int j=0; j<_clen; j++ )
					mb[0].getColumnMetadata(j).setMvValue(_mvMeta.get(j));
			if( _ndMeta != null )
				for( int j=0; j<_clen; j++ )
					mb[0].getColumnMetadata(j).setNumDistinct(Long.parseLong(_ndMeta.get(j)));
		}
	}
	
	/**
	 * 
	 */
	private static class BinaryBlockToCSVFunction implements FlatMapFunction<Tuple2<Long,FrameBlock>,String> 
	{
		private static final long serialVersionUID = 8020608184930291069L;

		private CSVFileFormatProperties _props = null;
		
		public BinaryBlockToCSVFunction(CSVFileFormatProperties props) {
			_props = props;
		}

		@Override
		public Iterable<String> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			Long ix = arg0._1();
			FrameBlock blk = arg0._2();
			
			ArrayList<String> ret = new ArrayList<String>();
			StringBuilder sb = new StringBuilder();
			
			//handle header information and frame meta data
			if( ix==1 ) {
				if( _props.hasHeader() ) {
					for(int j = 1; j <= blk.getNumColumns(); j++) {
						sb.append(blk.getColumnNames().get(j) 
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
			
			return ret;
		}
	}
	
	/////////////////////////////////
	// DataFrame-SPECIFIC FUNCTIONS
	
	private static class DataFrameToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Row,Long>>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = 2269315691094111843L;

		private long _clen = -1;
		private int _maxRowsPerBlock = -1;
		
		public DataFrameToBinaryBlockFunction(MatrixCharacteristics mc) {
			_clen = mc.getCols();
			_maxRowsPerBlock = Math.max((int) (FrameBlock.BUFFER_SIZE/_clen), 1);
		}
		
		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Row, Long>> arg0) throws Exception {
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<Tuple2<Long,FrameBlock>>();

			Long[] ix = new Long[1];
			FrameBlock[] mb = new FrameBlock[1];
			int iRowsInBlock = 0;
			
			while( arg0.hasNext() )
			{
				Tuple2<Row,Long> tmp = arg0.next();
				Row row = tmp._1();
				long rowix = tmp._2()+1;
				
				if( iRowsInBlock == 0 || iRowsInBlock == _maxRowsPerBlock) {
					if( iRowsInBlock == _maxRowsPerBlock )
						flushBlocksToList(ix, mb, ret);
					createBlocks(rowix, ix, mb, row);
					iRowsInBlock = 0;
				}
				
				//process row data
				Object[] parts = rowToObjectArray(row, (int)_clen, mb[0].getSchema());
				mb[0].appendRow(parts);
				iRowsInBlock++;
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret;
		}
		
		public Object[] rowToObjectArray(Row row, int _clen, List<ValueType> schema) throws Exception {
			Object[] ret = new Object[_clen];
			for(int i = 0; i < row.length(); i++)
				ret[i] = UtilFunctions.objectToObject(schema.get(i), row.get(i));
			for(int i=row.length(); i<_clen; i++)
				ret[i] = "";
			return ret;
		}

		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, Long[] ix, FrameBlock[] mb, Row row)
		{
			//compute row block index and number of column blocks
			ix[0] = new Long(rowix);
			
			List<String> columns = new ArrayList<String>();
			List<ValueType> schema = new ArrayList<ValueType>();
			for (StructField structType: row.schema().fields()) {
				columns.add(structType.name());
				if(structType.dataType() == DataTypes.DoubleType || structType.dataType() == DataTypes.FloatType)
					schema.add(ValueType.DOUBLE);
				else if(structType.dataType() == DataTypes.LongType || structType.dataType() == DataTypes.IntegerType)
					schema.add(ValueType.INT);
				else if(structType.dataType() == DataTypes.BooleanType)
					schema.add(ValueType.BOOLEAN);
				else
					schema.add(ValueType.STRING);
			}
			mb[0] = new FrameBlock(schema);
			mb[0].setColumnNames(columns);
		}
	}

	/**
	 * 
	 */
	private static class BinaryBlockToDataFrameFunction implements FlatMapFunction<Tuple2<Long,FrameBlock>,Row> 
	{
		private static final long serialVersionUID = 8093340778966667460L;
		
		@Override
		public Iterable<Row> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			FrameBlock blk = arg0._2();
			ArrayList<Row> ret = new ArrayList<Row>();

			//handle Frame block data
			Iterator<Object[]> iter = blk.getObjectRowIterator();
			while( iter.hasNext() )
				ret.add(RowFactory.create(iter.next().clone()));
				
			return ret;
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
		
		protected CellToBinaryBlockFunction(MatrixCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, FrameBlock.BUFFER_SIZE);
		}


		/**
		 * 
		 * @param rbuff
		 * @param ret
		 * @throws IOException 
		 * @throws DMLRuntimeException 
		 */
		protected void flushBufferToList( FrameReblockBuffer rbuff,  ArrayList<Tuple2<Long,FrameBlock>> ret ) 
			throws IOException, DMLRuntimeException
		{
			//temporary list of indexed matrix values to prevent library dependencies
			ArrayList<Pair<Long, FrameBlock>> rettmp = new ArrayList<Pair<Long, FrameBlock>>();
			rbuff.flushBufferToBinaryBlocks(rettmp);
			ret.addAll(SparkUtils.fromIndexedFrameBlock(rettmp));
		}
	}
	
	
	/**
	 * 
	 */
	private static class TextToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Text>,Long,FrameBlock> 
	{
		private static final long serialVersionUID = -2042208027876880588L;
		List<ValueType> _schema = null;
		
		protected TextToBinaryBlockFunction(MatrixCharacteristics mc, List<ValueType> schema ) {
			super(mc);
			_schema = schema;
		}

		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Iterator<Text> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<Tuple2<Long,FrameBlock>>();
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
				Object val = UtilFunctions.stringToObject(_schema.get((int)col-1), st.nextToken());
				
				//flush buffer if necessary
				if( rbuff.getSize() >= rbuff.getCapacity() )
					flushBufferToList(rbuff, ret);
				
				//add value to reblock buffer
				rbuff.appendCell(row, col, val);
			}
			
			//final flush buffer
			flushBufferToList(rbuff, ret);
		
			return ret;
		}
	}
	
	// MATRIX Block <---> Binary Block specific functions
	private static class MatrixToBinaryBlockFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,Long,FrameBlock>
	{
		private static final long serialVersionUID = 6205071301074768437L;

		private int _brlen = -1;
		private int _bclen = -1;
		private long _clen = -1;
		private int _maxRowsPerBlock = -1;
	
		
		public MatrixToBinaryBlockFunction(MatrixCharacteristics mc)
		{
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_clen = mc.getCols();
			_maxRowsPerBlock = Math.max((int) (FrameBlock.BUFFER_SIZE/_clen), 1);
		}

		@Override
		public Iterable<Tuple2<Long, FrameBlock>> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,FrameBlock>> ret = new ArrayList<Tuple2<Long,FrameBlock>>();

			MatrixIndexes matrixIndexes = arg0._1();
			MatrixBlock matrixBlock = arg0._2();
			
			//Frame Index (Row id, with base 1)
			Long rowix = new Long((matrixIndexes.getRowIndex()-1)*_brlen+1);

			//Global index within frame blocks
			long colixLow = (int)((matrixIndexes.getColumnIndex()-1)*_bclen+1);
			long colixHigh = Math.min(colixLow+matrixBlock.getNumColumns()-1, _clen);
			
			//Index within a local matrix block
			int iColLowMat = UtilFunctions.computeCellInBlock(colixLow, _bclen);
			int iColHighMat = UtilFunctions.computeCellInBlock(colixHigh, _bclen);

			FrameBlock tmpBlock = DataConverter.convertToFrameBlock(matrixBlock);

			int iRowLow = 0;	//Index within a local frame block
			while(iRowLow < matrixBlock.getNumRows()) {
				int iRowHigh = Math.min(iRowLow+_maxRowsPerBlock-1,  matrixBlock.getNumRows()-1);
				
				FrameBlock tmpBlock2 = null;
				//All rows from matrix block can fit into single frame block, no need for slicing 
				if(iRowLow == 0 && iRowHigh == matrixBlock.getNumRows()-1)
					tmpBlock2 = tmpBlock;
				else
					tmpBlock2 = tmpBlock.sliceOperations(iRowLow, iRowHigh, iColLowMat, iColHighMat, tmpBlock2);
				
				//If Matrix has only one column block, then simply assigns converted block to frame block
				if(colixLow == 0 && colixHigh == matrixBlock.getNumColumns()-1)
					ret.add(new Tuple2<Long, FrameBlock>(rowix+iRowLow, tmpBlock2));
				else
				{
					FrameBlock frameBlock = new FrameBlock((int)_clen, ValueType.STRING);
					frameBlock.ensureAllocatedColumns(iRowHigh-iRowLow+1);
					
					frameBlock.copy(0, iRowHigh-iRowLow, (int)colixLow-1, (int)colixHigh-1, tmpBlock2);
					ret.add(new Tuple2<Long, FrameBlock>(rowix+iRowLow, frameBlock));
				}
				iRowLow = iRowHigh+1;
			}
			return ret;
		}
	}

	/*
	 * This function supports if matrix has only one column block.
	 */
	private static class MatrixToBinaryBlockOneColumnBlockFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,Long,FrameBlock>
	{
		private static final long serialVersionUID = 3716019666116660815L;

		private int _brlen = -1;
		private int _bclen = -1;
		private long _clen = -1;
	
		
		public MatrixToBinaryBlockOneColumnBlockFunction(MatrixCharacteristics mc)
		{
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_clen = mc.getCols();
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			if(_clen > _bclen)
				throw new DMLRuntimeException("The input matrix has more than one column block, this function supports only one column block.");

			MatrixIndexes matrixIndexes = arg0._1();
			MatrixBlock matrixBlock = arg0._2();
			
			//Frame Index (Row id, with base 1)
			Long rowix = new Long((matrixIndexes.getRowIndex()-1)*_brlen+1);

			FrameBlock frameBlock = DataConverter.convertToFrameBlock(matrixBlock);
			return new Tuple2<Long, FrameBlock>(rowix, frameBlock);
		}
	}

	
	/**
	 * 
	 */
	private static class BinaryBlockToMatrixBlockFunction implements PairFlatMapFunction<Tuple2<Long,FrameBlock>,MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -2654986510471835933L;
		
		private MatrixCharacteristics _mcIn;
		private MatrixCharacteristics _mcOut;

		public BinaryBlockToMatrixBlockFunction(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut) {			
			_mcIn = mcIn;		//Frame Characteristics
			_mcOut = mcOut;		//Matrix Characteristics
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			long rowIndex = arg0._1();
			FrameBlock blk = arg0._2();
			
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			long rlen = _mcIn.getRows();
			long clen = _mcIn.getCols();
			int brlen = _mcOut.getRowsPerBlock();
			int bclen = _mcOut.getColsPerBlock();
			
			//slice aligned matrix blocks out of given frame block
			long rstartix = UtilFunctions.computeBlockIndex(rowIndex, brlen);
			long rendix = UtilFunctions.computeBlockIndex(rowIndex+blk.getNumRows()-1, brlen);
			long cendix = UtilFunctions.computeBlockIndex(blk.getNumColumns(), bclen);
			for( long rix=rstartix; rix<=rendix; rix++ ) { //for all row blocks
				long rpos = UtilFunctions.computeCellIndex(rix, brlen, 0);
				int lrlen = UtilFunctions.computeBlockSize(rlen, rix, brlen);
				int fix = (int)((rpos-rowIndex>=0) ? rpos-rowIndex : 0);
				int fix2 = (int)Math.min(rpos+lrlen-rowIndex-1,blk.getNumRows()-1);
				int mix = UtilFunctions.computeCellInBlock(rowIndex+fix, brlen);
				int mix2 = mix + (fix2-fix);
				for( long cix=1; cix<=cendix; cix++ ) { //for all column blocks
					long cpos = UtilFunctions.computeCellIndex(cix, bclen, 0);
					int lclen = UtilFunctions.computeBlockSize(clen, cix, bclen);
					MatrixBlock matrix = new MatrixBlock(lrlen, lclen, false);
					FrameBlock frame = blk.sliceOperations(fix, fix2, 
							(int)cpos-1, (int)cpos+lclen-2, new FrameBlock());
					MatrixBlock mframe = DataConverter.convertToMatrixBlock(frame);
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(rix, cix), 
							matrix.leftIndexingOperations(mframe, mix, mix2, 0, lclen-1, 
							new MatrixBlock(), UpdateType.INPLACE_PINNED)));
				}
			}

			return ret;
		}
	}
	
	//////////////////////////////////////
	// Common functions
	
	// Flushes current state of filled column blocks to output list.
	private static void flushBlocksToList( Long[] ix, FrameBlock[] mb, ArrayList<Tuple2<Long,FrameBlock>> ret ) 
		throws DMLRuntimeException
	{
		int len = ix.length;			
		for( int i=0; i<len; i++ )
			if( mb[i] != null ) {
				ret.add(new Tuple2<Long,FrameBlock>(ix[i],mb[i]));
			}	
	}
}
