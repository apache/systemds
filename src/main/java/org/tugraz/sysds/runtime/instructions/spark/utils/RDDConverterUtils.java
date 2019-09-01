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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.LongAccumulator;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.instructions.spark.data.SerLongWritable;
import org.tugraz.sysds.runtime.instructions.spark.data.SerText;
import org.tugraz.sysds.runtime.instructions.spark.functions.ConvertMatrixBlockToIJVLines;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesCSV;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesMM;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixCell;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.mapred.ReblockBuffer;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.FastStringTokenizer;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class RDDConverterUtils 
{
	public static final String DF_ID_COLUMN = "__INDEX";

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> textCellToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> input, DataCharacteristics mcOut, boolean outputEmptyBlocks, FileFormatPropertiesMM mmProps)
	{
		//convert textcell rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input.values()
				.mapPartitionsToPair(new TextToBinaryBlockFunction(mcOut, mmProps));

		//inject empty blocks (if necessary) 
		if( outputEmptyBlocks && mcOut.mightHaveEmptyBlocks() ) {
			out = out.union( 
				SparkUtils.getEmptyBlockRDD(sc, mcOut) );
		}
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey(out, false); 
		
		return out;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryCellToBinaryBlock(JavaSparkContext sc,
		JavaPairRDD<MatrixIndexes, MatrixCell> input, DataCharacteristics mcOut, boolean outputEmptyBlocks)
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
		out = RDDAggregateUtils.mergeByKey(out, false); 
		
		return out;
	}

	/**
	 * Converter from binary block rdd to rdd of labeled points. Note that the input needs to be 
	 * reblocked to satisfy the 'clen &lt;= blen' constraint.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @return JavaRDD of labeled points
	 */
	public static JavaRDD<LabeledPoint> binaryBlockToLabeledPoints(JavaPairRDD<MatrixIndexes, MatrixBlock> in) 
	{
		//convert indexed binary block input to collection of labeled points
		JavaRDD<LabeledPoint> pointrdd = in
				.values()
				.flatMap(new PrepareBinaryBlockFunction());
		
		return pointrdd;
	}

	public static JavaRDD<String> binaryBlockToTextCell(JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc) {
		return in.flatMap(new ConvertMatrixBlockToIJVLines(mc.getBlocksize()));
	}

	public static JavaRDD<String> binaryBlockToCsv(JavaPairRDD<MatrixIndexes,MatrixBlock> in, DataCharacteristics mcIn, FileFormatPropertiesCSV props, boolean strict)
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> input = in;
		
		//fast path without, general case with shuffle
		if( mcIn.getCols()>mcIn.getBlocksize() ) {
			//create row partitioned matrix
			input = input
					.flatMapToPair(new SliceBinaryBlockToRowsFunction(mcIn.getBlocksize()))
					.groupByKey()
					.mapToPair(new ConcatenateBlocksFunction(mcIn.getCols(), mcIn.getBlocksize()));	
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

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvToBinaryBlock(JavaSparkContext sc,
			JavaPairRDD<LongWritable, Text> input, DataCharacteristics mc,
			boolean hasHeader, String delim, boolean fill, double fillValue) {
		//determine unknown dimensions and sparsity if required
		if( !mc.dimsKnown(true) ) {
			LongAccumulator aNnz = sc.sc().longAccumulator("nnz");
			JavaRDD<String> tmp = input.values()
					.map(new CSVAnalysisFunction(aNnz, delim));
			long rlen = tmp.count() - (hasHeader ? 1 : 0);
			long clen = tmp.first().split(delim).length;
			long nnz = UtilFunctions.toLong(aNnz.value());
			mc.set(rlen, clen, mc.getBlocksize(), nnz);
		}
		
		//prepare csv w/ row indexes (sorted by filenames)
		JavaPairRDD<Text,Long> prepinput = input.values()
				.zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		boolean sparse = requiresSparseAllocation(prepinput, mc);
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(new CSVToBinaryBlockFunction(
						mc, sparse, hasHeader, delim, fill, fillValue));
		
		//aggregate partial matrix blocks (w/ preferred number of output 
		//partitions as the data is likely smaller in binary block format,
		//but also to bound the size of partitions for compressed inputs)
		int parts = SparkUtils.getNumPreferredPartitions(mc, out);
		return RDDAggregateUtils.mergeByKey(out, parts, false); 
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvToBinaryBlock(JavaSparkContext sc,
			JavaRDD<String> input, DataCharacteristics mcOut,
			boolean hasHeader, String delim, boolean fill, double fillValue) 
	{
		//convert string rdd to serializable longwritable/text
		JavaPairRDD<LongWritable, Text> prepinput =
				input.mapToPair(new StringToSerTextFunction());
		
		//convert to binary block
		return csvToBinaryBlock(sc, prepinput, mcOut, hasHeader, delim, fill, fillValue);
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
		Dataset<Row> df, DataCharacteristics mc, boolean containsID, boolean isVector)
	{
		//determine unknown dimensions and sparsity if required
		if( !mc.dimsKnown(true) ) {
			LongAccumulator aNnz = sc.sc().longAccumulator("nnz");
			JavaRDD<Row> tmp = df.javaRDD().map(new DataFrameAnalysisFunction(aNnz, containsID, isVector));
			long rlen = tmp.count();
			long clen = !isVector ? df.columns().length - (containsID?1:0) : 
					((Vector) tmp.first().get(containsID?1:0)).size();
			long nnz = UtilFunctions.toLong(aNnz.value());
			mc.set(rlen, clen, mc.getBlocksize(), nnz);
		}
		
		//ensure valid blocksizes
		if( mc.getBlocksize()<=1 || mc.getBlocksize()<=1 ) {
			mc.setBlockSize(ConfigurationManager.getBlocksize());
		}
		
		//construct or reuse row ids
		JavaPairRDD<Row, Long> prepinput = containsID ?
				df.javaRDD().mapToPair(new DataFrameExtractIDFunction(
					df.schema().fieldIndex(DF_ID_COLUMN))) :
				df.javaRDD().zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		boolean sparse = requiresSparseAllocation(prepinput, mc);
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(
					new DataFrameToBinaryBlockFunction(mc, sparse, containsID, isVector));
		
		//aggregate partial matrix blocks (w/ preferred number of output 
		//partitions as the data is likely smaller in binary block format,
		//but also to bound the size of partitions for compressed inputs)
		int parts = SparkUtils.getNumPreferredPartitions(mc, out);
		return RDDAggregateUtils.mergeByKey(out, parts, false); 
	}

	public static Dataset<Row> binaryBlockToDataFrame(SparkSession sparkSession,
	                                                  JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc, boolean toVector)
	{
		if( !mc.colsKnown() )
			throw new RuntimeException("Number of columns needed to convert binary block to data frame.");
		
		//slice blocks into rows, align and convert into data frame rows
		JavaRDD<Row> rowsRDD = in
			.flatMapToPair(new SliceBinaryBlockToRowsFunction(mc.getBlocksize()))
			.groupByKey().map(new ConvertRowBlocksToRows((int)mc.getCols(), mc.getBlocksize(), toVector));
		
		//create data frame schema
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(DF_ID_COLUMN, DataTypes.DoubleType, false));
		if( toVector )
			fields.add(DataTypes.createStructField("C1", new VectorUDT(), false));
		else { // row
			for(int i = 1; i <= mc.getCols(); i++)
				fields.add(DataTypes.createStructField("C"+i, DataTypes.DoubleType, false));
		}
		
		//rdd to data frame conversion
		return sparkSession.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
	}

	@Deprecated
	public static Dataset<Row> binaryBlockToDataFrame(SQLContext sqlContext,
	                                                  JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc, boolean toVector)
	{
		SparkSession sparkSession = sqlContext.sparkSession();
		return binaryBlockToDataFrame(sparkSession, in, mc, toVector);
	}

	/**
	 * Converts a libsvm text input file into two binary block matrices for features 
	 * and labels, and saves these to the specified output files. This call also deletes 
	 * existing files at the specified output locations, as well as determines and 
	 * writes the meta data files of both output matrices. 
	 * <p>
	 * Note: We use {@code org.apache.spark.mllib.util.MLUtils.loadLibSVMFile} for parsing 
	 * the libsvm input files in order to ensure consistency with Spark.
	 * 
	 * @param sc java spark context
	 * @param pathIn path to libsvm input file
	 * @param pathX path to binary block output file of features
	 * @param pathY path to binary block output file of labels
	 * @param mcOutX matrix characteristics of output matrix X
	 */
	public static void libsvmToBinaryBlock(JavaSparkContext sc, String pathIn, 
			String pathX, String pathY, DataCharacteristics mcOutX)
	{
		if( !mcOutX.dimsKnown() )
			throw new DMLRuntimeException("Matrix characteristics "
				+ "required to convert sparse input representation.");
		try {
			//cleanup existing output files
			HDFSTool.deleteFileIfExistOnHDFS(pathX);
			HDFSTool.deleteFileIfExistOnHDFS(pathY);
			
			//convert libsvm to labeled points
			int numFeatures = (int) mcOutX.getCols();
			int numPartitions = SparkUtils.getNumPreferredPartitions(mcOutX, null);
			JavaRDD<org.apache.spark.mllib.regression.LabeledPoint> lpoints = 
					MLUtils.loadLibSVMFile(sc.sc(), pathIn, numFeatures, numPartitions).toJavaRDD();
			
			//append row index and best-effort caching to avoid repeated text parsing
			JavaPairRDD<org.apache.spark.mllib.regression.LabeledPoint,Long> ilpoints = 
					lpoints.zipWithIndex().persist(StorageLevel.MEMORY_AND_DISK()); 
			
			//extract labels and convert to binary block
			DataCharacteristics mc1 = new MatrixCharacteristics(mcOutX.getRows(), 1, mcOutX.getBlocksize(), -1);
			LongAccumulator aNnz1 = sc.sc().longAccumulator("nnz");
			JavaPairRDD<MatrixIndexes,MatrixBlock> out1 = ilpoints
					.mapPartitionsToPair(new LabeledPointToBinaryBlockFunction(mc1, true, aNnz1));
			int numPartitions2 = SparkUtils.getNumPreferredPartitions(mc1, null);
			out1 = RDDAggregateUtils.mergeByKey(out1, numPartitions2, false);
			out1.saveAsHadoopFile(pathY, MatrixIndexes.class, MatrixBlock.class, SequenceFileOutputFormat.class);
			mc1.setNonZeros(aNnz1.value()); //update nnz after triggered save
			HDFSTool.writeMetaDataFile(pathY+".mtd", ValueType.FP64, mc1, OutputInfo.BinaryBlockOutputInfo);
			
			//extract data and convert to binary block
			DataCharacteristics mc2 = new MatrixCharacteristics(mcOutX.getRows(), mcOutX.getCols(), mcOutX.getBlocksize(), -1);
			LongAccumulator aNnz2 = sc.sc().longAccumulator("nnz");
			JavaPairRDD<MatrixIndexes,MatrixBlock> out2 = ilpoints
					.mapPartitionsToPair(new LabeledPointToBinaryBlockFunction(mc2, false, aNnz2));
			out2 = RDDAggregateUtils.mergeByKey(out2, numPartitions, false);
			out2.saveAsHadoopFile(pathX, MatrixIndexes.class, MatrixBlock.class, SequenceFileOutputFormat.class);
			mc2.setNonZeros(aNnz2.value()); //update nnz after triggered save
			HDFSTool.writeMetaDataFile(pathX+".mtd", ValueType.FP64, mc2, OutputInfo.BinaryBlockOutputInfo);
			
			//asynchronous cleanup of cached intermediates
			ilpoints.unpersist(false);
		}
		catch(IOException ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	public static JavaPairRDD<LongWritable, Text> stringToSerializableText(JavaPairRDD<Long,String> in)
	{
		return in.mapToPair(new TextToSerTextFunction());
	}

	private static boolean requiresSparseAllocation(JavaPairRDD<?,?> in, DataCharacteristics mc) {
		//if nnz unknown or sparse, pick the robust sparse representation
		if( !mc.nnzKnown() || (mc.nnzKnown() && MatrixBlock.evalSparseFormatInMemory(
			mc.getRows(), mc.getCols(), mc.getNonZeros())) ) {
			return true;
		}
		
		//if dense evaluate expected rows per partition to handle wide matrices
		//(pick sparse representation if fraction of rows per block less than sparse theshold)
		double datasize = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc);
		double rowsize = OptimizerUtils.estimatePartitionedSizeExactSparsity(1, mc.getCols(),
			mc.getNumRowBlocks(), Math.ceil((double)mc.getNonZeros()/mc.getRows()));
		double partsize = Math.ceil(datasize/in.getNumPartitions());
		double blksz = Math.min(mc.getRows(), mc.getBlocksize());
		return partsize/rowsize/blksz < MatrixBlock.SPARSITY_TURN_POINT;
	}

	private static int countNnz(Object vect, boolean isVector, int off) {
		if( isVector ) //note: numNonzeros scans entries but handles sparse/dense
			return ((Vector) vect).numNonzeros();
		else 
			return countNnz(vect, isVector, off, ((Row)vect).length());
	}

	/**
	 * Count the number of non-zeros for a subrange of the given row.
	 * 
	 * @param vect row object (row of basic types or row including a vector)
	 * @param isVector if the row includes a vector
	 * @param pos physical position 
	 * @param cu logical upper column index (exclusive) 
	 * @return number of non-zeros.
	 */
	private static int countNnz(Object vect, boolean isVector, int pos, int cu ) {
		int lnnz = 0;
		if( isVector ) {
			if( vect instanceof DenseVector ) {
				DenseVector vec = (DenseVector) vect;
				for( int i=pos; i<cu; i++ )
					lnnz += (vec.apply(i) != 0) ? 1 : 0;
			}
			else if( vect instanceof SparseVector ) {
				SparseVector vec = (SparseVector) vect;
				int alen = vec.numActives();
				int[] aix = vec.indices();
				double[] avals = vec.values();
				for( int i=pos; i<alen && aix[i]<cu; i++ )
					lnnz += (avals[i] != 0) ? 1 : 0;
			}
		}
		else { //row
			Row row = (Row) vect;
			for( int i=pos; i<cu; i++ )
				lnnz += UtilFunctions.isNonZero(row.get(i)) ? 1 : 0;
		}
		return lnnz;
	}
	
	private static Vector createVector(MatrixBlock row) {
		if( row.isEmptyBlock(false) ) //EMPTY SPARSE ROW
			return Vectors.sparse(row.getNumColumns(), new int[0], new double[0]);
		else if( row.isInSparseFormat() ) //SPARSE ROW
			return Vectors.sparse(row.getNumColumns(), 
					row.getSparseBlock().indexes(0), row.getSparseBlock().values(0));
		else // DENSE ROW
			return Vectors.dense(row.getDenseBlockValues());
	}
	
	/////////////////////////////////
	// BINARYBLOCK-SPECIFIC FUNCTIONS

	/**
	 * This function converts a binary block input (&lt;X,y&gt;) into mllib's labeled points. Note that
	 * this function requires prior reblocking if the number of columns is larger than the column
	 * block size. 
	 */
	private static class PrepareBinaryBlockFunction implements FlatMapFunction<MatrixBlock, LabeledPoint> 
	{
		private static final long serialVersionUID = -6590259914203201585L;

		@Override
		public Iterator<LabeledPoint> call(MatrixBlock arg0) 
			throws Exception 
		{
			ArrayList<LabeledPoint> ret = new ArrayList<>();
			for( int i=0; i<arg0.getNumRows(); i++ ) {
				MatrixBlock tmp = arg0.slice(i, i, 0, arg0.getNumColumns()-2, new MatrixBlock());
				ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), createVector(tmp)));
			}
			
			return ret.iterator();
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
		protected int _blen = -1;
		
		protected CellToBinaryBlockFunction(DataCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_blen = mc.getBlocksize();
			_blen = mc.getBlocksize();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, BUFFER_SIZE);
		}

		protected void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws IOException, DMLRuntimeException
		{
			rbuff.flushBufferToBinaryBlocks().stream() // prevent library dependencies
				.map(b -> SparkUtils.fromIndexedMatrixBlock(b)).forEach(b -> ret.add(b));
		}
	}

	private static class TextToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Text>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4907483236186747224L;

		private final FileFormatPropertiesMM _mmProps;
		
		protected TextToBinaryBlockFunction(DataCharacteristics mc, FileFormatPropertiesMM mmProps) {
			super(mc);
			_mmProps = mmProps;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Text> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _blen);
			FastStringTokenizer st = new FastStringTokenizer(' ');
			boolean first = false;
			
			while( arg0.hasNext() ) {
				//get input string (ignore matrix market comments as well as
				//first row which indicates meta data, i.e., <nrow> <ncol> <nnz>)
				String strVal = arg0.next().toString();
				if( strVal.startsWith("%") ) {
					first = true;
					continue;
				}
				else if (first) {
					first = false;
					continue;
				}
				
				//parse input ijv triple
				st.reset( strVal.toString() ); //reinit tokenizer
				long row = st.nextLong();
				long col = st.nextLong();
				double val = (_mmProps == null) ? st.nextDouble() : 
					_mmProps.isPatternField() ? 1 : _mmProps.isIntField() ? st.nextLong() : st.nextDouble();
				
				//flush buffer if necessary
				if( rbuff.getSize() >= rbuff.getCapacity() )
					flushBufferToList(rbuff, ret);
				
				//add value to reblock buffer
				rbuff.appendCell(row, col, val);
				if( _mmProps != null && _mmProps.isSymmetric() && row!=col )
					rbuff.appendCell(col, row, val);
			}
			
			//final flush buffer
			flushBufferToList(rbuff, ret);
		
			return ret.iterator();
		}
	}

	private static class TextToSerTextFunction implements PairFunction<Tuple2<Long,String>,LongWritable,Text> 
	{
		private static final long serialVersionUID = 2286037080400222528L;

		@Override
		public Tuple2<LongWritable, Text> call(Tuple2<Long, String> arg0) 
			throws Exception 
		{
			SerLongWritable slarg = new SerLongWritable(arg0._1());
			SerText starg = new SerText(arg0._2());	
			return new Tuple2<>(slarg, starg);
		}
	}

	private static class StringToSerTextFunction implements PairFunction<String, LongWritable, Text> 
	{
		private static final long serialVersionUID = 2286037080400222528L;

		@Override
		public Tuple2<LongWritable, Text> call(String arg0) 
			throws Exception 
		{
			SerLongWritable slarg = new SerLongWritable(1L);
			SerText starg = new SerText(arg0);
			return new Tuple2<>(slarg, starg);
		}
	}
	
	/////////////////////////////////
	// BINARYCELL-SPECIFIC FUNCTIONS

	public static class BinaryCellToBinaryBlockFunction extends CellToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixCell>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 3928810989462198243L;

		public BinaryCellToBinaryBlockFunction(DataCharacteristics mc) {
			super(mc);
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes,MatrixCell>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _blen);
			
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
		
			return ret.iterator();
		}
	}
	
	/////////////////////////////////
	// CSV-SPECIFIC FUNCTIONS

	private static class CSVAnalysisFunction implements Function<Text,String> 
	{
		private static final long serialVersionUID = 2310303223289674477L;

		private LongAccumulator _aNnz = null;
		private String _delim = null;
		
		public CSVAnalysisFunction( LongAccumulator aNnz, String delim )
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
			int lnnz = IOUtilFunctions.countNnz(cols);
			
			//update counters
			_aNnz.add( lnnz );
			
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
		private int _blen = -1;
		private double _sparsity = 1.0;
		private boolean _sparse = false;
		private boolean _header = false;
		private String _delim = null;
		private boolean _fill = false;
		private double _fillValue = 0;
		
		public CSVToBinaryBlockFunction(DataCharacteristics mc, boolean sparse, boolean hasHeader, String delim, boolean fill, double fillValue)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_blen = mc.getBlocksize();
			_blen = mc.getBlocksize();
			_sparsity = OptimizerUtils.getSparsity(mc);
			_sparse = sparse && (!fill || fillValue==0);
			_header = hasHeader;
			_delim = delim;
			_fill = fill;
			_fillValue = fillValue;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Text,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();

			int ncblks = (int)Math.ceil((double)_clen/_blen);
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
				
				long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _blen);
			
				//create new blocks for entire row
				if( ix[0] == null || ix[0].getRowIndex() != rix ) {
					if( ix[0] !=null )
						flushBlocksToList(ix, mb, ret);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					createBlocks(rowix, (int)len, ix, mb);
				}
				
				//process row data
				String[] parts = IOUtilFunctions.split(row, _delim);
				boolean emptyFound = false;
				for( int cix=1, pix=0; cix<=ncblks; cix++ ) 
				{
					int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);
					if( mb[cix-1].isInSparseFormat() ) {
						//allocate row once (avoid re-allocations)
						int lnnz = IOUtilFunctions.countNnz(parts, pix, lclen);
						mb[cix-1].getSparseBlock().allocate(pos, lnnz);
					}
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
		
			return ret.iterator();
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb)
		{
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
			int ncblks = (int)Math.ceil((double)_clen/_blen);
			
			//create all column blocks (assume dense since csv is dense text format)
			for( int cix=1; cix<=ncblks; cix++ ) {
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);				
				ix[cix-1] = new MatrixIndexes(rix, cix);
				mb[cix-1] = new MatrixBlock(lrlen, lclen, _sparse, (int)(lrlen*lclen*_sparsity));
				mb[cix-1].allocateBlock();
			}
		}
		
		// Flushes current state of filled column blocks to output list.
		private static void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) {
			int len = ix.length;
			for( int i=0; i<len; i++ )
				if( mb[i] != null ) {
					ret.add(new Tuple2<>(ix[i],mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}
		}
	}

	private static class LabeledPointToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<org.apache.spark.mllib.regression.LabeledPoint,Long>>,MatrixIndexes,MatrixBlock> 
	{	
		private static final long serialVersionUID = 2290124693964816276L;
		
		private final long _rlen;
		private final long _clen;
		private final int _blen;
		private final boolean _sparseX;
		private final boolean _labels;
		private final LongAccumulator _aNnz;
		
		public LabeledPointToBinaryBlockFunction(DataCharacteristics mc, boolean labels, LongAccumulator aNnz) {
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_blen = mc.getBlocksize();
			_sparseX = MatrixBlock.evalSparseFormatInMemory(
					mc.getRows(), mc.getCols(), mc.getNonZeros());
			_labels = labels;
			_aNnz = aNnz;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<org.apache.spark.mllib.regression.LabeledPoint,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();

			int ncblks = (int)Math.ceil((double)_clen/_blen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			
			while( arg0.hasNext() )
			{
				Tuple2<org.apache.spark.mllib.regression.LabeledPoint,Long> tmp = arg0.next();
				org.apache.spark.mllib.regression.LabeledPoint row = tmp._1();
				boolean lsparse = _sparseX || (!_labels && 
						row.features() instanceof org.apache.spark.mllib.linalg.SparseVector);
				long rowix = tmp._2() + 1;
				
				long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _blen);
			
				//create new blocks for entire row
				if( ix[0] == null || ix[0].getRowIndex() != rix ) {
					if( ix[0] !=null )
						flushBlocksToList(ix, mb, ret);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					createBlocks(rowix, (int)len, ix, mb, lsparse);
				}
				
				//process row data
				if( _labels ) {
					double val = row.label();
					mb[0].appendValue(pos, 0, val);
					_aNnz.add((val != 0) ? 1 : 0);
				}
				else { //features
					int lnnz = row.features().numNonzeros();
					if( row.features() instanceof org.apache.spark.mllib.linalg.SparseVector )
					{
						org.apache.spark.mllib.linalg.SparseVector srow = 
								(org.apache.spark.mllib.linalg.SparseVector) row.features();
						for( int k=0; k<lnnz; k++ ) {
							int gix = srow.indices()[k]+1;
							int cix = (int)UtilFunctions.computeBlockIndex(gix, _blen);
							int j = UtilFunctions.computeCellInBlock(gix, _blen);
							mb[cix-1].appendValue(pos, j, srow.values()[k]);
						}
					}
					else { //dense
						for( int cix=1, pix=0; cix<=ncblks; cix++ ) {
							int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);
							for( int j=0; j<lclen; j++ )
								mb[cix-1].appendValue(pos, j, row.features().apply(pix++));
						}
					}
					_aNnz.add(lnnz);
				}
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret.iterator();
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb, boolean lsparse)
		{
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
			int ncblks = (int)Math.ceil((double)_clen/_blen);
			
			//create all column blocks (assume dense since csv is dense text format)
			for( int cix=1; cix<=ncblks; cix++ ) {
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);
				ix[cix-1] = new MatrixIndexes(rix, cix);
				mb[cix-1] = new MatrixBlock(lrlen, lclen, lsparse);
				mb[cix-1].allocateBlock();
			}
		}
		
		// Flushes current state of filled column blocks to output list.
		private static void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) {
			int len = ix.length;
			for( int i=0; i<len; i++ )
				if( mb[i] != null ) {
					ret.add(new Tuple2<>(ix[i],mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}
		}
	}
	
	private static class BinaryBlockToCSVFunction implements FlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,String> 
	{
		private static final long serialVersionUID = 1891768410987528573L;

		private FileFormatPropertiesCSV _props = null;
		
		public BinaryBlockToCSVFunction(FileFormatPropertiesCSV props) {
			_props = props;
		}

		@Override
		public Iterator<String> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception 
		{
			MatrixIndexes ix = arg0._1();
			MatrixBlock blk = arg0._2();
			
			ArrayList<String> ret = new ArrayList<>();
			
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
			
			return ret.iterator();
		}
	}

	private static class SliceBinaryBlockToRowsFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,Long,Tuple2<Long,MatrixBlock>> 
	{
		private static final long serialVersionUID = 7192024840710093114L;
		
		private int _blen = -1;
		
		public SliceBinaryBlockToRowsFunction(int blen) {
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<Long,Tuple2<Long,MatrixBlock>>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<Long,Tuple2<Long,MatrixBlock>>> ret = new ArrayList<>();
			MatrixIndexes ix = arg0._1();
			MatrixBlock blk = arg0._2();
			for( int i=0; i<blk.getNumRows(); i++ ) {
				MatrixBlock tmpBlk = blk.slice(i, i);
				long rix = UtilFunctions.computeCellIndex(ix.getRowIndex(), _blen, i);
				ret.add(new Tuple2<>(rix, new Tuple2<>(ix.getColumnIndex(),tmpBlk)));
			}
			return ret.iterator();
		}
		
	}

	private static class ConcatenateBlocksFunction implements PairFunction<Tuple2<Long, Iterable<Tuple2<Long,MatrixBlock>>>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = -7879603125149650097L;
		
		private long _clen = -1;
		private int _blen = -1;
		private int _ncblks = -1;
		
		public ConcatenateBlocksFunction(long clen, int blen) {
			_clen = clen;
			_blen = blen;
			_ncblks = (int)Math.ceil((double)clen/blen);
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
			for( int i=0; i<_ncblks; i++ )
				out.copy(0, 0, i*_blen, (int)Math.min((i+1)*_blen, _clen)-1, tmpBlks[i], false);
			out.recomputeNonZeros();
			//output row block
			return new Tuple2<>(new MatrixIndexes(rowIndex, 1),out);
		}
	}

	/////////////////////////////////
	// DATAFRAME-SPECIFIC FUNCTIONS

	private static class DataFrameToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Row,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 653447740362447236L;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _blen = -1;
		private double _sparsity = 1.0;
		private boolean _sparse = false;
		private boolean _containsID;
		private boolean _isVector;
		
		public DataFrameToBinaryBlockFunction(DataCharacteristics mc, boolean sparse, boolean containsID, boolean isVector) {
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_blen = mc.getBlocksize();
			_sparsity = OptimizerUtils.getSparsity(mc);
			_sparse = sparse;
			_containsID = containsID;
			_isVector = isVector;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Row, Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			int ncblks = (int)Math.ceil((double)_clen/_blen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			
			while( arg0.hasNext() )
			{
				Tuple2<Row,Long> tmp = arg0.next();
				long rowix = tmp._2() + 1;
				
				long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _blen);
			
				//create new blocks for entire row
				if( ix[0] == null || ix[0].getRowIndex() != rix ) {
					if( ix[0] !=null )
						flushBlocksToList(ix, mb, ret);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					createBlocks(rowix, (int)len, ix, mb);
				}
				
				//process row data
				int off = _containsID ? 1 : 0;
				Object obj = _isVector ? tmp._1().get(off) : tmp._1();
				for( int cix=1, pix=_isVector?0:off; cix<=ncblks; cix++ ) {
					int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);
					int cu = (int) Math.min(_clen, cix*_blen) + (_isVector?0:off);
					//allocate sparse row once (avoid re-allocations)
					if( mb[cix-1].isInSparseFormat() ) {
						int lnnz = countNnz(obj, _isVector, pix, cu);
						mb[cix-1].getSparseBlock().allocate(pos, lnnz);
					}
					//append data to matrix blocks
					if( _isVector ) {
						Vector vect = (Vector) obj;
						if( vect instanceof SparseVector ) {
							SparseVector svect = (SparseVector) vect;
							int[] svectIx = svect.indices();
							while( pix<svectIx.length && svectIx[pix]<cu ) {
								int j = UtilFunctions.computeCellInBlock(svectIx[pix]+1, _blen);
								mb[cix-1].appendValue(pos, j, svect.values()[pix++]);
							}
						}
						else { //dense
							for( int j=0; j<lclen; j++ )
								mb[cix-1].appendValue(pos, j, vect.apply(pix++));
						}
					}
					else { //row
						Row row = (Row) obj;
						for( int j=0; j<lclen; j++ )
							mb[cix-1].appendValue(pos, j, UtilFunctions.getDouble(row.get(pix++)));
					}
				}
			}
		
			//flush last blocks
			flushBlocksToList(ix, mb, ret);
		
			return ret.iterator();
		}
		
		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb)
		{
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _blen);
			int ncblks = (int)Math.ceil((double)_clen/_blen);
			
			//create all column blocks (assume dense since csv is dense text format)
			for( int cix=1; cix<=ncblks; cix++ ) {
				int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _blen);
				ix[cix-1] = new MatrixIndexes(rix, cix);
				mb[cix-1] = new MatrixBlock(lrlen, lclen, _sparse,(int)(lrlen*lclen*_sparsity));
				mb[cix-1].allocateBlock();
			}
		}
		
		// Flushes current state of filled column blocks to output list.
		private static void flushBlocksToList( MatrixIndexes[] ix, MatrixBlock[] mb, ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) {
			int len = ix.length;
			for( int i=0; i<len; i++ )
				if( mb[i] != null ) {
					ret.add(new Tuple2<>(ix[i],mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}
		}
	}

	private static class DataFrameAnalysisFunction implements Function<Row,Row>  
	{	
		private static final long serialVersionUID = 5705371332119770215L;
		
		private LongAccumulator _aNnz = null;
		private boolean _containsID;
		private boolean _isVector;
		
		public DataFrameAnalysisFunction( LongAccumulator aNnz, boolean containsID, boolean isVector) {
			_aNnz = aNnz;
			_containsID = containsID;
			_isVector = isVector;
		}

		@Override
		public Row call(Row arg0) throws Exception {
			//determine number of non-zeros of row
			int off = _containsID ? 1 : 0;
			Object vect = _isVector ? arg0.get(off) : arg0;
			int lnnz = countNnz(vect, _isVector, off);
			
			//update counters
			_aNnz.add( lnnz );
			return arg0;
		}
	}

	public static class DataFrameExtractIDFunction implements PairFunction<Row, Row,Long> 
	{
		private static final long serialVersionUID = 7438855241666363433L;

		private int _index = -1;
		
		public DataFrameExtractIDFunction(int index) {
			_index = index;
		}

		@Override
		public Tuple2<Row, Long> call(Row arg0) throws Exception {
			//extract 1-based IDs and convert to 0-based positions
			long id = UtilFunctions.toLong(UtilFunctions.getDouble(arg0.get(_index)));
			if( id <= 0 ) {
				throw new DMLRuntimeException("ID Column '" + DF_ID_COLUMN 
						+ "' expected to be 1-based, but found value: "+id);
			}
			return new Tuple2<>(arg0, id-1);
		}
	}

	private static class ConvertRowBlocksToRows implements Function<Tuple2<Long, Iterable<Tuple2<Long, MatrixBlock>>>, Row> {
		
		private static final long serialVersionUID = 4441184411670316972L;
		
		private int _clen;
		private int _blen;
		private boolean _toVector;
		
		public ConvertRowBlocksToRows(int clen, int blen, boolean toVector) {
			_clen = clen;
			_blen = blen;
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
				if( _clen <= _blen ) { //single block
					row[1] = createVector(arg0._2().iterator().next()._2());
				}
				else { //multiple column blocks
					double[] tmp = new double[_clen];
					for(Tuple2<Long, MatrixBlock> kv : arg0._2()) {
						int cl = (kv._1().intValue()-1)*_blen;
						MatrixBlock mb = kv._2();
						DataConverter.copyToDoubleVector(mb, tmp, cl);
					}
					row[1] = Vectors.dense(tmp);
				}
			}
			else {
				for(Tuple2<Long, MatrixBlock> kv : arg0._2()) {
					int cl = (kv._1().intValue()-1)*_blen;
					MatrixBlock mb = kv._2();
					for( int j=0; j<mb.getNumColumns(); j++ )
						row[cl+j+1] = mb.quickGetValue(0, j);
				}
			}
			
			return RowFactory.create(row);
		}
	}
}
