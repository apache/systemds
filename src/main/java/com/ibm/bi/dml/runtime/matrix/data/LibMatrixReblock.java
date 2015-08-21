package com.ibm.bi.dml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
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
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertTextLineToBinaryCellFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CountLines;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;

/**
 * This class allows commonly used reblock-related functions for Spark runtime. This should be used 
 * by implementors of APIs and various Spark instructions.
 *
 */
public class LibMatrixReblock {

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
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> textRDDToBinaryBlockRDD(
			JavaPairRDD<LongWritable, Text> lines,
			MatrixCharacteristics mcIn, MatrixCharacteristics mcOut, JavaSparkContext sc,
			int brlen, int bclen, boolean outputEmptyBlocks) 
			throws DMLRuntimeException  {
		JavaRDD<String> textLines = lines.values().map(new ConvertTextToString());
		return textRDDToBinaryBlockRDD(textLines, mcIn, mcOut, sc, brlen, bclen, outputEmptyBlocks);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> textRDDToBinaryBlockRDD(
			JavaRDD<String> lines, 
			MatrixCharacteristics mcIn, MatrixCharacteristics mcOut, JavaSparkContext sc,
			int brlen, int bclen, boolean outputEmptyBlocks) 
			throws DMLRuntimeException  {
		long numRows = -1;
		long numColumns = -1;
		if(!mcOut.dimsKnown() && !mcIn.dimsKnown()) {
			throw new DMLRuntimeException("Unknown dimensions in reblock instruction for text format");
		}
		else if(mcIn.dimsKnown()) {
			numRows = mcIn.getRows();
			numColumns = mcIn.getCols();
		}
		else {
			numRows = mcOut.getRows();
			numColumns = mcOut.getCols();
		}
		
		if(numRows <= 0 || numColumns <= 0) {
			throw new DMLRuntimeException("Error: Incorrect input dimensions:" + numRows + "," +  numColumns); 
		}
		
		JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = 
				lines.mapToPair(new ConvertTextLineToBinaryCellFunction(numRows, numColumns, brlen, bclen))
				.filter(new DropEmptyBinaryCells());
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = binaryCellRDDToBinaryBlockRDD(binaryCells, mcIn, mcOut, sc, brlen, bclen, outputEmptyBlocks);
		return out;
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> binaryCellRDDToBinaryBlockRDD(
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells,
			MatrixCharacteristics mc, MatrixCharacteristics mcOut, 
			JavaSparkContext sc, int brlen, int bclen, boolean outputEmptyBlocks) 
		throws DMLRuntimeException 
	{	
		long numRows = -1;
		long numColumns = -1;
		if(!mcOut.dimsKnown() && !mc.dimsKnown()) {
			throw new DMLRuntimeException("Unknown dimensions while reblock into binary cell format");
		}
		else if(mc.dimsKnown()) {
			numRows = mc.getRows();
			numColumns = mc.getCols();
		}
		else {
			numRows = mcOut.getRows();
			numColumns = mcOut.getCols();
		}
		
		if(numRows <= 0 || numColumns <= 0) {
			throw new DMLRuntimeException("Error: Incorrect input dimensions:" + numRows + "," +  numColumns); 
		}
		
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
						.mapToPair(new ConvertALToBinaryBlockFunction(brlen, bclen, numRows, numColumns));		
		// ----------------------------------------------------------------------------
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithEmptyBlocks = null;
		if(outputEmptyBlocks) {
			binaryBlocksWithEmptyBlocks = SparkUtils.getRDDWithEmptyBlocks(sc, 
					binaryBlocksWithoutEmptyBlocks, numRows, numColumns, brlen, bclen);
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
	
}
