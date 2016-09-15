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
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

import scala.Tuple2;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.data.SerLongWritable;
import org.apache.sysml.runtime.instructions.spark.data.SerText;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertMatrixBlockToIJVLines;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.ReblockBuffer;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.UtilFunctions;

public class RDDConverterUtils 
{
	public static final String DF_ID_COLUMN = "__INDEX";
	
	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param outputEmptyBlocks
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> textCellToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> input, MatrixCharacteristics mcOut, boolean outputEmptyBlocks) 
		throws DMLRuntimeException  
	{
 		//convert textcell rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input.values()
				.mapPartitionsToPair(new TextToBinaryBlockFunction(mcOut));

		//inject empty blocks (if necessary) 
		if( outputEmptyBlocks && mcOut.mightHaveEmptyBlocks() ) {
			out = out.union( 
				SparkUtils.getEmptyBlockRDD(sc, mcOut) );
		}
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}

	/**
	 * 
	 * @param sc
	 * @param input
	 * @param mcOut
	 * @param outputEmptyBlocks
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryCellToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixCell> input, MatrixCharacteristics mcOut, boolean outputEmptyBlocks) 
		throws DMLRuntimeException 
	{	
 		//convert binarycell rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input
				.mapPartitionsToPair(new BinaryCellToBinaryBlockFunction(mcOut));

		//inject empty blocks (if necessary) 
		if( outputEmptyBlocks && mcOut.mightHaveEmptyBlocks() ) {
			out = out.union( 
				SparkUtils.getEmptyBlockRDD(sc, mcOut) );
		}
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}

	/**
	 * Converter from binary block rdd to rdd of labeled points. Note that the input needs to be 
	 * reblocked to satisfy the 'clen <= bclen' constraint.
	 * 
	 * @param in
	 * @return
	 */
	public static JavaRDD<LabeledPoint> binaryBlockToLabeledPoints(JavaPairRDD<MatrixIndexes, MatrixBlock> in) 
	{
		//convert indexed binary block input to collection of labeled points
		JavaRDD<LabeledPoint> pointrdd = in
				.values()
				.flatMap(new PrepareBinaryBlockFunction());
		
		return pointrdd;
	}

	/**
	 * 
	 * @param in
	 * @param mc
	 * @return
	 */
	public static JavaRDD<String> binaryBlockToTextCell(JavaPairRDD<MatrixIndexes, MatrixBlock> in, MatrixCharacteristics mc) {
		return in.flatMap(new ConvertMatrixBlockToIJVLines(
				mc.getRowsPerBlock(), mc.getColsPerBlock()));
	}
	
	/**
	 * 
	 * @param in
	 * @param mcIn
	 * @param props
	 * @param strict
	 * @return
	 */
	public static JavaRDD<String> binaryBlockToCsv(JavaPairRDD<MatrixIndexes,MatrixBlock> in, MatrixCharacteristics mcIn, CSVFileFormatProperties props, boolean strict)
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> input = in;
		
		//fast path without, general case with shuffle
		if( mcIn.getCols()>mcIn.getColsPerBlock() ) {
			//create row partitioned matrix
			input = input
					.flatMapToPair(new SliceBinaryBlockToRowsFunction(mcIn.getRowsPerBlock()))
					.groupByKey()
					.mapToPair(new ConcatenateBlocksFunction(mcIn.getCols(), mcIn.getColsPerBlock()));	
		}
		
		//sort if required (on blocks/rows)
		if( strict ) {
			input = input.sortByKey(true);
		}
		
		//convert binary block to csv (from blocks/rows)
		JavaRDD<String> out = input
				.flatMap(new BinaryBlockToCSVFunction(props));
	
		return out;
	}
	
	/**
	 * 
	 * @param sc
	 * @param lines
	 * @param mcOut
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> input, MatrixCharacteristics mcOut, 
			boolean hasHeader, String delim, boolean fill, double fillValue) 
		throws DMLRuntimeException 
	{
		//determine unknown dimensions and sparsity if required
		if( !mcOut.dimsKnown(true) ) {
			Accumulator<Double> aNnz = sc.accumulator(0L);
			JavaRDD<String> tmp = input.values()
					.map(new CSVAnalysisFunction(aNnz, delim));
			long rlen = tmp.count() - (hasHeader ? 1 : 0);
			long clen = tmp.first().split(delim).length;
			long nnz = UtilFunctions.toLong(aNnz.value());
			mcOut.set(rlen, clen, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), nnz);
		}
		
		//prepare csv w/ row indexes (sorted by filenames)
		JavaPairRDD<Text,Long> prepinput = input.values()
				.zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(
					new CSVToBinaryBlockFunction(mcOut, hasHeader, delim, fill, fillValue));
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}
	
	/**
	 * Example usage:
	 * <pre><code>
	 * import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils
	 * import org.apache.sysml.runtime.matrix.MatrixCharacteristics
	 * import org.apache.spark.api.java.JavaSparkContext
	 * val A = sc.textFile("ranA.csv")
	 * val Amc = new MatrixCharacteristics
	 * val Abin = RDDConverterUtils.csvToBinaryBlock(new JavaSparkContext(sc), A, Amc, false, ",", false, 0)
	 * </code></pre>
	 * 
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvToBinaryBlock(JavaSparkContext sc,
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
	 * @param sc
	 * @param df
	 * @param mcOut
	 * @param containsID
	 * @param isVector
	 * @param columns
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame df, MatrixCharacteristics mc, boolean containsID, boolean isVector) 
	{
		//determine unknown dimensions and sparsity if required
		if( !mc.dimsKnown(true) ) {
			Accumulator<Double> aNnz = sc.accumulator(0L);
			JavaRDD<Row> tmp = df.javaRDD().map(new DataFrameAnalysisFunction(aNnz, containsID, isVector));
			long rlen = tmp.count();
			long clen = !isVector ? df.columns().length - (containsID?1:0) : 
					((Vector) tmp.first().get(containsID?1:0)).size();
			long nnz = UtilFunctions.toLong(aNnz.value());
			mc.set(rlen, clen, mc.getRowsPerBlock(), mc.getColsPerBlock(), nnz);
		}
		
		//ensure valid blocksizes
		if( mc.getRowsPerBlock()<=1 || mc.getColsPerBlock()<=1 ) {
			mc.setBlockSize(ConfigurationManager.getBlocksize());
		}
		
		JavaPairRDD<Row, Long> prepinput = containsID ?
				df.javaRDD().mapToPair(new DataFrameExtractIDFunction()) :
				df.javaRDD().zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(
					new DataFrameToBinaryBlockFunction(mc, containsID, isVector));
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}
	
	/**
	 * 
	 * @param sqlContext
	 * @param in
	 * @param mc
	 * @param toVector
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DataFrame binaryBlockToDataFrame(SQLContext sqlctx, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in, MatrixCharacteristics mc, boolean toVector)  
	{
		if( !mc.colsKnown() )
			throw new RuntimeException("Number of columns needed to convert binary block to data frame.");
		
		//slice blocks into rows, align and convert into data frame rows
		JavaRDD<Row> rowsRDD = in
			.flatMapToPair(new SliceBinaryBlockToRowsFunction(mc.getRowsPerBlock()))
			.groupByKey().map(new ConvertRowBlocksToRows((int)mc.getCols(), mc.getColsPerBlock(), toVector));
		
		//create data frame schema
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField(DF_ID_COLUMN, DataTypes.DoubleType, false));
		if( toVector )
			fields.add(DataTypes.createStructField("C1", new VectorUDT(), false));
		else { // row
			for(int i = 1; i <= mc.getCols(); i++)
				fields.add(DataTypes.createStructField("C"+i, DataTypes.DoubleType, false));
		}
		
		//rdd to data frame conversion
		return sqlctx.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<LongWritable, Text> stringToSerializableText(JavaPairRDD<Long,String> in)
	{
		return in.mapToPair(new TextToSerTextFunction());
	}

	
	/////////////////////////////////
	// BINARYBLOCK-SPECIFIC FUNCTIONS

	/**
	 * This function converts a binary block input (<X,y>) into mllib's labeled points. Note that
	 * this function requires prior reblocking if the number of columns is larger than the column
	 * block size. 
	 */
	private static class PrepareBinaryBlockFunction implements FlatMapFunction<MatrixBlock, LabeledPoint> 
	{
		private static final long serialVersionUID = -6590259914203201585L;

		@Override
		public Iterable<LabeledPoint> call(MatrixBlock arg0) 
			throws Exception 
		{
			ArrayList<LabeledPoint> ret = new ArrayList<LabeledPoint>();
			for( int i=0; i<arg0.getNumRows(); i++ )
			{
				MatrixBlock tmp = arg0.sliceOperations(i, i, 0, arg0.getNumColumns()-2, new MatrixBlock());
				double[] data = DataConverter.convertToDoubleVector(tmp);
				if( tmp.isEmptyBlock(false) ) //EMPTY SPARSE ROW
				{
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.sparse(0, new int[0], new double[0])));
				}
				else if( tmp.isInSparseFormat() ) //SPARSE ROW
				{
					SparseBlock sblock = tmp.getSparseBlock();
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), 
							Vectors.sparse(sblock.size(0), sblock.indexes(0), sblock.values(0))));
				}
				else // DENSE ROW
				{
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.dense(data)));
				}
			}
			
			return ret;
		}
	}
	
	/////////////////////////////////
	// TEXTCELL-SPECIFIC FUNCTIONS
	
	private static abstract class CellToBinaryBlockFunction implements Serializable
	{
		private static final long serialVersionUID = 4205331295408335933L;
		
		//internal buffer size (aligned w/ default matrix block size)
		protected static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
		protected int _bufflen = -1;
		
		protected long _rlen = -1;
		protected long _clen = -1;
		protected int _brlen = -1;
		protected int _bclen = -1;
		
		protected CellToBinaryBlockFunction(MatrixCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, BUFFER_SIZE);
		}


		/**
		 * 
		 * @param rbuff
		 * @param ret
		 * @throws IOException 
		 * @throws DMLRuntimeException 
		 */
		protected void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws IOException, DMLRuntimeException
		{
			//temporary list of indexed matrix values to prevent library dependencies
			ArrayList<IndexedMatrixValue> rettmp = new ArrayList<IndexedMatrixValue>();					
			rbuff.flushBufferToBinaryBlocks(rettmp);
			ret.addAll(SparkUtils.fromIndexedMatrixBlock(rettmp));
		}
	}
	
	/**
	 * 
	 */
	private static class TextToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Text>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4907483236186747224L;

		protected TextToBinaryBlockFunction(MatrixCharacteristics mc) {
			super(mc);
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Text> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);
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
				double val = st.nextDouble();
				
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
	
	/**
	 * 
	 */
	private static class TextToSerTextFunction implements PairFunction<Tuple2<Long,String>,LongWritable,Text> 
	{
		private static final long serialVersionUID = 2286037080400222528L;

		@Override
		public Tuple2<LongWritable, Text> call(Tuple2<Long, String> arg0) 
			throws Exception 
		{
			SerLongWritable slarg = new SerLongWritable(arg0._1());
			SerText starg = new SerText(arg0._2());			
			return new Tuple2<LongWritable,Text>(slarg, starg);
		}
	}
	
	/**
	 * 
	 */
	private static class StringToSerTextFunction implements PairFunction<String, LongWritable, Text> 
	{
		private static final long serialVersionUID = 2286037080400222528L;

		@Override
		public Tuple2<LongWritable, Text> call(String arg0) 
			throws Exception 
		{
			SerLongWritable slarg = new SerLongWritable(1L);
			SerText starg = new SerText(arg0);			
			return new Tuple2<LongWritable,Text>(slarg, starg);
		}
	}
	
	/////////////////////////////////
	// BINARYCELL-SPECIFIC FUNCTIONS

	private static class BinaryCellToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixCell>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 3928810989462198243L;

		protected BinaryCellToBinaryBlockFunction(MatrixCharacteristics mc) {
			super(mc);
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes,MatrixCell>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);
			
			while( arg0.hasNext() )
			{
				//unpack the binary cell input
				Tuple2<MatrixIndexes,MatrixCell> tmp = arg0.next();
				
				//parse input ijv triple
				long row = tmp._1().getRowIndex();
				long col = tmp._1().getColumnIndex();
				double val = tmp._2().getValue();
				
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
	
	/////////////////////////////////
	// CSV-SPECIFIC FUNCTIONS

	/**
	 * 
	 */
	private static class CSVAnalysisFunction implements Function<Text,String> 
	{
		private static final long serialVersionUID = 2310303223289674477L;

		private Accumulator<Double> _aNnz = null;
		private String _delim = null;
		
		public CSVAnalysisFunction( Accumulator<Double> aNnz, String delim )
		{
			_aNnz = aNnz;
			_delim = delim;
		}
		
		@Override
		public String call(Text v1) 
			throws Exception 
		{
			//parse input line
			String line = v1.toString();
			String[] cols = IOUtilFunctions.split(line, _delim);
			
			//determine number of non-zeros of row (w/o string parsing)
			long lnnz = 0;
			for( String col : cols ) {
				lnnz += (!col.isEmpty() && !col.equals("0") 
						&& !col.equals("0.0")) ? 1 : 0;
			}
			
			//update counters
			_aNnz.add( (double)lnnz );
			
			return line;
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
	private static class CSVToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Text,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -4948430402942717043L;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		private double _sparsity = 1.0;
		private boolean _sparse = false;
		private boolean _header = false;
		private String _delim = null;
		private boolean _fill = false;
		private double _fillValue = 0;
		
		public CSVToBinaryBlockFunction(MatrixCharacteristics mc, boolean hasHeader, String delim, boolean fill, double fillValue)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_sparsity = OptimizerUtils.getSparsity(mc);
			_sparse = mc.nnzKnown() && MatrixBlock.evalSparseFormatInMemory(mc.getRows(), 
					mc.getCols(), mc.getNonZeros()) && (!fill || fillValue==0);
			_header = hasHeader;
			_delim = delim;
			_fill = fill;
			_fillValue = fillValue;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Text,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();

			int ncblks = (int)Math.ceil((double)_clen/_bclen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			
			while( arg0.hasNext() )
			{
				Tuple2<Text,Long> tmp = arg0.next();
				String row = tmp._1().toString();
				long rowix = tmp._2() + (_header ? 0 : 1);
				
				//skip existing header
				if( _header && rowix == 0  ) 
					continue;
				
				long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _brlen);
			
				//create new blocks for entire row
				if( ix[0] == null || ix[0].getRowIndex() != rix ) {
					if( ix[0] !=null )
						flushBlocksToList(ix, mb, ret);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					createBlocks(rowix, (int)len, ix, mb);
				}
				
				//process row data
				String[] parts = IOUtilFunctions.split(row, _delim);
				boolean emptyFound = false;
				for( int cix=1, pix=0; cix<=ncblks; cix++ ) 
				{
					int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
					for( int j=0; j<lclen; j++ ) {
						String part = parts[pix++];
						emptyFound |= part.isEmpty() && !_fill;
						double val = (part.isEmpty() && _fill) ?
								_fillValue : Double.parseDouble(part);
						mb[cix-1].appendValue(pos, j, val);
					}	
				}
		
				//sanity check empty cells filled w/ values
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(row, _fill, emptyFound);
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret;
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb)
		{
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
			int ncblks = (int)Math.ceil((double)_clen/_bclen);
			
			//create all column blocks (assume dense since csv is dense text format)
			for( int cix=1; cix<=ncblks; cix++ ) {
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
				ix[cix-1] = new MatrixIndexes(rix, cix);
				mb[cix-1] = new MatrixBlock(lrlen, lclen, _sparse, (int)(lrlen*lclen*_sparsity));		
			}
		}
		
		// Flushes current state of filled column blocks to output list.
		private void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws DMLRuntimeException
		{
			int len = ix.length;			
			for( int i=0; i<len; i++ )
				if( mb[i] != null ) {
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix[i],mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}	
		}
	}
	
	/**
	 * 
	 */
	private static class BinaryBlockToCSVFunction implements FlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,String> 
	{
		private static final long serialVersionUID = 1891768410987528573L;

		private CSVFileFormatProperties _props = null;
		
		public BinaryBlockToCSVFunction(CSVFileFormatProperties props) {
			_props = props;
		}

		@Override
		public Iterable<String> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception 
		{
			MatrixIndexes ix = arg0._1();
			MatrixBlock blk = arg0._2();
			
			ArrayList<String> ret = new ArrayList<String>();
			
			//handle header information
			if(_props.hasHeader() && ix.getRowIndex()==1 ) {
				StringBuilder sb = new StringBuilder();
	    		for(int j = 1; j < blk.getNumColumns(); j++) {
	    			if(j != 1)
	    				sb.append(_props.getDelim());
	    			sb.append("C" + j);
	    		}
    			ret.add(sb.toString());
	    	}
		
			//handle matrix block data
			StringBuilder sb = new StringBuilder();
    		for(int i=0; i<blk.getNumRows(); i++) {
    			for(int j=0; j<blk.getNumColumns(); j++) {
	    			if(j != 0)
	    				sb.append(_props.getDelim());
	    			double val = blk.quickGetValue(i, j);
	    			if(!(_props.isSparse() && val == 0))
	    				sb.append(val);
				}
	    		ret.add(sb.toString());
	    		sb.setLength(0); //reset
    		}
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class SliceBinaryBlockToRowsFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,Long,Tuple2<Long,MatrixBlock>> 
	{
		private static final long serialVersionUID = 7192024840710093114L;
		
		private int _brlen = -1;
		
		public SliceBinaryBlockToRowsFunction(int brlen) {
			_brlen = brlen;
		}
		
		@Override
		public Iterable<Tuple2<Long,Tuple2<Long,MatrixBlock>>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,Tuple2<Long,MatrixBlock>>> ret = 
					new ArrayList<Tuple2<Long,Tuple2<Long,MatrixBlock>>>();
			
			MatrixIndexes ix = arg0._1();
			MatrixBlock blk = arg0._2();
			
			for( int i=0; i<blk.getNumRows(); i++ ) {
				MatrixBlock tmpBlk = blk.sliceOperations(i, i, 0, blk.getNumColumns()-1, new MatrixBlock());
				long rix = UtilFunctions.computeCellIndex(ix.getRowIndex(), _brlen, i);
				ret.add(new Tuple2<Long,Tuple2<Long,MatrixBlock>>(rix, 
						new Tuple2<Long,MatrixBlock>(ix.getColumnIndex(),tmpBlk)));
			}
			
			return ret;
		}
		
	}
	
	/**
	 * 
	 */
	private static class ConcatenateBlocksFunction implements PairFunction<Tuple2<Long, Iterable<Tuple2<Long,MatrixBlock>>>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = -7879603125149650097L;
		
		private long _clen = -1;
		private int _bclen = -1;
		private int _ncblks = -1;
		
		public ConcatenateBlocksFunction(long clen, int bclen) {
			_clen = clen;
			_bclen = bclen;
			_ncblks = (int)Math.ceil((double)clen/bclen);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<Long,Iterable<Tuple2<Long, MatrixBlock>>> arg0)
			throws Exception 
		{
			long rowIndex = arg0._1();
			MatrixBlock[] tmpBlks = new MatrixBlock[_ncblks];
			
			//collect and sort input blocks
			Iterator<Tuple2<Long, MatrixBlock>> iter = arg0._2().iterator();
			while( iter.hasNext() ) {
				Tuple2<Long, MatrixBlock> entry = iter.next();
				tmpBlks[entry._1().intValue()-1] = entry._2();
			}
		
			//concatenate blocks
			MatrixBlock out = new MatrixBlock(1,(int)_clen, tmpBlks[0].isInSparseFormat());
			for( int i=0; i<_ncblks; i++ ) {				
				out.copy(0, 0, i*_bclen, (int)Math.min((i+1)*_bclen, _clen)-1, tmpBlks[i], false);				
			}
			out.recomputeNonZeros();
			
			//output row block
			return new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(rowIndex, 1),out);
		}		
	}

	/////////////////////////////////
	// DATAFRAME-SPECIFIC FUNCTIONS

	/**
	 * 
	 */
	private static class DataFrameToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Row,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 653447740362447236L;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		private boolean _sparse = false;
		private boolean _containsID;
		private boolean _isVector;
		
		public DataFrameToBinaryBlockFunction(MatrixCharacteristics mc, boolean containsID, boolean isVector) {
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_sparse = mc.nnzKnown() && MatrixBlock.evalSparseFormatInMemory(
					mc.getRows(), mc.getCols(), mc.getNonZeros());
			_containsID = containsID;
			_isVector = isVector;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Row, Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			
			int ncblks = (int)Math.ceil((double)_clen/_bclen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			
			while( arg0.hasNext() )
			{
				Tuple2<Row,Long> tmp = arg0.next();
				long rowix = tmp._2() + 1;
				
				long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _brlen);
			
				//create new blocks for entire row
				if( ix[0] == null || ix[0].getRowIndex() != rix ) {
					if( ix[0] !=null )
						flushBlocksToList(ix, mb, ret);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					createBlocks(rowix, (int)len, ix, mb);
				}
				
				//process row data
				int off = _containsID ? 1: 0;
				if( _isVector ) {
					Vector vect = (Vector) tmp._1().get(off);
					for( int cix=1, pix=0; cix<=ncblks; cix++ ) {
						int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
						for( int j=0; j<lclen; j++ )
							mb[cix-1].appendValue(pos, j, vect.apply(pix++));
					}	
				}
				else { //row
					Row row = tmp._1();
					for( int cix=1, pix=off; cix<=ncblks; cix++ ) {
						int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
						for( int j=0; j<lclen; j++ )
							mb[cix-1].appendValue(pos, j, UtilFunctions.getDouble(row.get(pix++)));
					}
				}
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret;		
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb)
		{
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
			int ncblks = (int)Math.ceil((double)_clen/_bclen);
			
			//create all column blocks (assume dense since csv is dense text format)
			for( int cix=1; cix<=ncblks; cix++ ) {
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
				ix[cix-1] = new MatrixIndexes(rix, cix);
				mb[cix-1] = new MatrixBlock(lrlen, lclen, _sparse);		
			}
		}
		
		// Flushes current state of filled column blocks to output list.
		private void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws DMLRuntimeException
		{
			int len = ix.length;			
			for( int i=0; i<len; i++ )
				if( mb[i] != null ) {
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix[i],mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}	
		}
	}
	
	/**
	 * 
	 */
	private static class DataFrameAnalysisFunction implements Function<Row,Row>  
	{	
		private static final long serialVersionUID = 5705371332119770215L;
		
		private Accumulator<Double> _aNnz = null;
		private boolean _containsID;
		private boolean _isVector;
		
		public DataFrameAnalysisFunction( Accumulator<Double> aNnz, boolean containsID, boolean isVector) {
			_aNnz = aNnz;
			_containsID = containsID;
			_isVector = isVector;
		}

		@Override
		public Row call(Row arg0) throws Exception {
			//determine number of non-zeros of row
			int off = _containsID ? 1 : 0;
			long lnnz = 0;
			if( _isVector ) {
				//note: numNonzeros scans entries but handles sparse/dense
				Vector vec = (Vector) arg0.get(off);
				lnnz += vec.numNonzeros();
			}
			else { //row
				for(int i=off; i<arg0.length(); i++)
					lnnz += UtilFunctions.isNonZero(arg0.get(i)) ? 1 : 0;
			}
		
			//update counters
			_aNnz.add( (double)lnnz );
			return arg0;
		}
	}

	/**
	 * 
	 */
	protected static class DataFrameExtractIDFunction implements PairFunction<Row, Row,Long> 
	{
		private static final long serialVersionUID = 7438855241666363433L;

		@Override
		public Tuple2<Row, Long> call(Row arg0) throws Exception {
			//extract 1-based IDs and convert to 0-based positions
			long id = UtilFunctions.toLong(UtilFunctions.getDouble(arg0.get(0)));
			if( id <= 0 ) {
				throw new DMLRuntimeException("ID Column '" + DF_ID_COLUMN 
						+ "' expected to be 1-based, but found value: "+id);
			}
			return new Tuple2<Row,Long>(arg0, id-1);
		}
	}
	
	/**
	 * 
	 */
	private static class ConvertRowBlocksToRows implements Function<Tuple2<Long, Iterable<Tuple2<Long, MatrixBlock>>>, Row> {
		
		private static final long serialVersionUID = 4441184411670316972L;
		
		private int _clen;
		private int _bclen;
		private boolean _toVector;
		
		public ConvertRowBlocksToRows(int clen, int bclen, boolean toVector) {
			_clen = clen;
			_bclen = bclen;
			_toVector = toVector;
		}

		@Override
		public Row call(Tuple2<Long, Iterable<Tuple2<Long, MatrixBlock>>> arg0)
			throws Exception 
		{
			Object[] row = new Object[_toVector ? 2 : _clen+1];
			row[0] = (double) arg0._1(); //row index
			
			//copy block data into target row
			if( _toVector ) {
				double[] tmp = new double[_clen];
				for(Tuple2<Long, MatrixBlock> kv : arg0._2()) {
					int cl = (kv._1().intValue()-1)*_bclen;
					MatrixBlock mb = kv._2();
					DataConverter.copyToDoubleVector(mb, tmp, cl);
				}
				row[1] = new DenseVector(tmp);
			}
			else {
				for(Tuple2<Long, MatrixBlock> kv : arg0._2()) {
					int cl = (kv._1().intValue()-1)*_bclen;
					MatrixBlock mb = kv._2();
					for( int j=0; j<mb.getNumColumns(); j++ )
						row[cl+j+1] = mb.quickGetValue(0, j);
				}
			}
			
			return RowFactory.create(row);
		}
	}
}
