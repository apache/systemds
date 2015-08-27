/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

import scala.Tuple2;

import com.ibm.bi.dml.api.MLOutput.ConvertDoubleArrayToRows;
import com.ibm.bi.dml.api.MLOutput.ProjectRows;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.spark.data.CountLinesInfo;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertALToBinaryBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertCSVLinesToMatrixBlocks;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertRowToCSVString;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertTextToString;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CountLines;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.ReblockBuffer;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

public class RDDConverterUtils 
{
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
	
	public static JavaRDD<String> dataFrameToCSVRDD(DataFrame df, boolean containsID) throws DMLRuntimeException {
		if(containsID) {
			// Uncomment this when we move to Spark 1.4.0 or higher 
			// df = df.sort("ID").drop("ID");
			throw new DMLRuntimeException("Ignoring ID is not supported yet");
		}
		
		JavaRDD<String> rdd = null;
		if(df != null && df.javaRDD() != null) {
			rdd = df.javaRDD().map(new ConvertRowToCSVString());
		}
		else {
			throw new DMLRuntimeException("Unsupported DataFrame as it is not backed by rdd");
		}
		return rdd;
	}
	
	public static DataFrame binaryBlockToDataFrame(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockRDD, 
			MatrixCharacteristics mc, SQLContext sqlContext) throws DMLRuntimeException {
		long rlen = mc.getRows(); long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock(); int bclen = mc.getColsPerBlock();
		
		// Very expensive operation here: groupByKey (where number of keys might be too large)
		JavaRDD<Row> rowsRDD = binaryBlockRDD.flatMapToPair(new ProjectRows(rlen, clen, brlen, bclen))
				.groupByKey().map(new ConvertDoubleArrayToRows(clen, bclen));
		
		int numColumns = (int) clen;
		if(numColumns <= 0) {
			// numColumns = rowsRDD.first().length() - 1; // Ugly, so instead prefer to throw
			throw new DMLRuntimeException("Output dimensions unknown after executing the script and hence cannot create the dataframe");
		}
		
		List<StructField> fields = new ArrayList<StructField>();
		// LongTypes throw an error: java.lang.Double incompatible with java.lang.Long
		fields.add(DataTypes.createStructField("ID", DataTypes.DoubleType, false)); 
		for(int i = 1; i <= numColumns; i++) {
			fields.add(DataTypes.createStructField("C" + i, DataTypes.DoubleType, false));
		}
		
		// This will cause infinite recursion due to bug in Spark
		// https://issues.apache.org/jira/browse/SPARK-6999
		// return sqlContext.createDataFrame(rowsRDD, colNames); // where ArrayList<String> colNames
		return sqlContext.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
	}
			
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvRDDToBinaryBlockRDD(
			JavaPairRDD<LongWritable, Text> lines, 
			MatrixCharacteristics mcOut, JavaSparkContext sc, int brlen, int bclen,
			boolean hasHeader, String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		JavaRDD<String> csvLines = lines.values().map(new ConvertTextToString());
		return csvRDDToBinaryBlockRDD(csvLines, mcOut, sc, brlen, bclen,
				hasHeader, delim, fill, missingValue);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> csvRDDToBinaryBlockRDD(
			JavaRDD<String> csvLines, 
			MatrixCharacteristics mcOut, JavaSparkContext sc, int brlen, int bclen,
			boolean hasHeader, String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		HashMap<Integer, Long> rowOffsets = getRowOffsets(csvLines, delim, mcOut, brlen, bclen);
		
		// When size of offset is large, broadcast is much better than task-serialization. 
		Broadcast<HashMap<Integer, Long>> broadcastRowOffset = sc.broadcast(rowOffsets);
					
		JavaPairRDD<MatrixIndexes, MatrixBlock> chunks = JavaPairRDD.fromJavaRDD(csvLines.mapPartitionsWithIndex(
						new ConvertCSVLinesToMatrixBlocks(broadcastRowOffset, 
								mcOut.getRows(), mcOut.getCols(), mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), 
								hasHeader, delim, fill, missingValue), true));

		// Merge chunks according to their block index
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDAggregateUtils.mergeByKey(chunks);
		return out;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryCellRDDToBinaryBlockRDD(JavaSparkContext sc,
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells, MatrixCharacteristics mcOut, boolean outputEmptyBlocks) 
		throws DMLRuntimeException 
	{	
		// TODO: Investigate whether binaryCells.persist() will help here or not
		
		// ----------------------------------------------------------------------------
		// Now merge binary cells into binary blocks
		// Here you provide three "extremely light-weight" functions (that ignores sparsity):
		// 1. cell -> ArrayList (AL)
		// 2. (AL, cell) -> AL
		// 3. (AL, AL) -> AL
		// Then you convert the final AL -> binary blocks (here you take into account the sparsity).
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithoutEmptyBlocks =
				binaryCells.combineByKey(
						new ConvertCellToALFunction(), 
						new AddCellToALFunction(), 
						new MergeALFunction())
						.mapToPair(new ConvertALToBinaryBlockFunction(mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), mcOut.getRows(), mcOut.getCols()));		
		// ----------------------------------------------------------------------------
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithEmptyBlocks = null;
		if(outputEmptyBlocks) {
			binaryBlocksWithEmptyBlocks = SparkUtils.getRDDWithEmptyBlocks(sc, 
					binaryBlocksWithoutEmptyBlocks, mcOut.getRows(), mcOut.getCols(), mcOut.getRowsPerBlock(), mcOut.getColsPerBlock());
		}
		else {
			binaryBlocksWithEmptyBlocks = binaryBlocksWithoutEmptyBlocks;
		}
		
		return binaryBlocksWithEmptyBlocks;
	}
	
	private static HashMap<Integer, Long> getRowOffsets(JavaRDD<String> csvLines, String delim,
			MatrixCharacteristics mcOut, int brlen, int bclen) throws DMLRuntimeException {
		// Start by counting the number of lines in each partition.
		List<Tuple2<Integer, CountLinesInfo>> linesPerPartition = 
				JavaPairRDD.fromJavaRDD(
						csvLines.mapPartitionsWithIndex(new CountLines(delim), true)
				)
				.sortByKey().collect();
		
		if(linesPerPartition.size() == 0) {
			throw new DMLRuntimeException("Expected atleast one partition for the CSV input file");
		}
		
		// Compute the offset of the first line in the each partition.
		// This code assumes that partitions are numbered in order, but does
		// not assume that
		// partition numbers are contiguous
		HashMap<Integer, Long> rowOffsets = new HashMap<Integer, Long>();
		long expectedNumColumns = -1;
		long numRows = 0;
		rowOffsets.put(linesPerPartition.get(0)._1, 0L);
		
		int prevPartNo = linesPerPartition.get(0)._1;
		for (int i = 1; i < linesPerPartition.size(); i++) {
			Integer partNo = linesPerPartition.get(i)._1;
			Long prevOffset = rowOffsets.get(prevPartNo);
			CountLinesInfo info = linesPerPartition.get(i - 1)._2;
			long curOffset = prevOffset + info.getNumLines();
			expectedNumColumns = Math.max(expectedNumColumns, info.getExpectedNumColumns());
			numRows += info.getNumLines();
			rowOffsets.put(partNo, curOffset);
			prevPartNo = partNo;
		}
		CountLinesInfo lastInfo = linesPerPartition.get(linesPerPartition.size() - 1)._2;
		expectedNumColumns = Math.max(expectedNumColumns, lastInfo.getExpectedNumColumns());
		numRows += lastInfo.getNumLines();
		
		if(mcOut.dimsKnown() && (mcOut.getRows() != numRows || mcOut.getCols() != expectedNumColumns)) {
			throw new DMLRuntimeException("Incorrect number of dimensions in csv reblock");
		}
		else if(!mcOut.dimsKnown()) {
			mcOut.set(numRows, expectedNumColumns, mcOut.getRowsPerBlock(), brlen, bclen);
		}
		
		return rowOffsets;
	}
	
		
	
	
	// ====================================================================================================
	// Three functions passed to combineByKey
	
	public static class ConvertCellToALFunction implements Function<MatrixCell, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -2458721762929481811L;
		@Override
		public ArrayList<MatrixCell> call(MatrixCell cell) throws Exception {
			ArrayList<MatrixCell> retVal = new ArrayList<MatrixCell>();
			if(cell.getValue() != 0)
				retVal.add(cell);
			return retVal;
		}	
	}
	
	public static class AddCellToALFunction implements Function2<ArrayList<MatrixCell>, MatrixCell, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -4680403897867388102L;
		@Override
		public ArrayList<MatrixCell> call(ArrayList<MatrixCell> al, MatrixCell cell) throws Exception {
			al.add(cell);
			return al;
		}	
	}
	
	public static class MergeALFunction implements Function2<ArrayList<MatrixCell>, ArrayList<MatrixCell>, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -8117257799807223694L;
		@Override
		public ArrayList<MatrixCell> call(ArrayList<MatrixCell> al1, ArrayList<MatrixCell> al2) throws Exception {
			al1.addAll(al2);
			return al1;
		}	
	}
	// ====================================================================================================
	
	// This function gets called to check whether to drop binary cell corresponding to header of Matrix market format
	public static class DropEmptyBinaryCells implements Function<Tuple2<MatrixIndexes,MatrixCell>, Boolean> {
		private static final long serialVersionUID = -3672377410407066396L;
		
		@Override
		public Boolean call(Tuple2<MatrixIndexes, MatrixCell> arg0) throws Exception {
			if(arg0._1.getRowIndex() == -1) {
				return false; // Header cell for MatrixMarket format
			}
			else if(arg0._2.getValue() == 0) {
				return false; // empty cell: can be dropped as MatrixBlock can handle sparsity
			}
			return true;
		}
		
	}
	
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
					SparseRow row = tmp.getSparseRows()[0]; 
					int rlen = row.size();
					int[] rix = row.getIndexContainer();
					double[] rvals = row.getValueContainer();
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.sparse(rlen, rix, rvals)));
				}
				else // DENSE ROW
				{
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.dense(data)));
				}
			}
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class TextToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Text>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4907483236186747224L;
		
		//internal buffer size (aligned w/ default matrix block size)
		private static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
		private int _bufflen = -1;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public TextToBinaryBlockFunction(MatrixCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, BUFFER_SIZE);
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

		/**
		 * 
		 * @param rbuff
		 * @param ret
		 * @throws IOException 
		 * @throws DMLRuntimeException 
		 */
		private void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws IOException, DMLRuntimeException
		{
			//temporary list of indexed matrix values to prevent library dependencies
			ArrayList<IndexedMatrixValue> rettmp = new ArrayList<IndexedMatrixValue>();					
			rbuff.flushBufferToBinaryBlocks(rettmp);
			ret.addAll(SparkUtils.fromIndexedMatrixBlock(rettmp));
		}
	}
}
