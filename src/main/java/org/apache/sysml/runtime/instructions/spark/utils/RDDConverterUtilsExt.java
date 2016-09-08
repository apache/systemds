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
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import org.apache.hadoop.io.Text;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import scala.Tuple2;

import org.apache.sysml.api.MLOutput.ConvertDoubleArrayToRows;
import org.apache.sysml.api.MLOutput.ProjectRows;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertMatrixBlockToIJVLines;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.ReblockBuffer;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * NOTE: These are experimental converter utils. Once thoroughly tested, they
 * can be moved to RDDConverterUtils.
 */
@SuppressWarnings("unused")
public class RDDConverterUtilsExt 
{
	public enum RDDConverterTypes {
		TEXT_TO_MATRIX_CELL, 
		MATRIXENTRY_TO_MATRIXCELL,
		TEXT_TO_DOUBLEARR, 
		ROW_TO_DOUBLEARR, 
		VECTOR_TO_DOUBLEARR
	}
	
	
	/**
	 * Example usage:
	 * <pre><code>
	 * import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt
	 * import org.apache.sysml.runtime.matrix.MatrixCharacteristics
	 * import org.apache.spark.api.java.JavaSparkContext
	 * import org.apache.spark.mllib.linalg.distributed.MatrixEntry
	 * import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
	 * val matRDD = sc.textFile("ratings.text").map(_.split(" ")).map(x => new MatrixEntry(x(0).toLong, x(1).toLong, x(2).toDouble)).filter(_.value != 0).cache
	 * require(matRDD.filter(x => x.i == 0 || x.j == 0).count == 0, "Expected 1-based ratings file")
	 * val nnz = matRDD.count
	 * val numRows = matRDD.map(_.i).max
	 * val numCols = matRDD.map(_.j).max
	 * val coordinateMatrix = new CoordinateMatrix(matRDD, numRows, numCols)
	 * val mc = new MatrixCharacteristics(numRows, numCols, 1000, 1000, nnz)
	 * val binBlocks = RDDConverterUtilsExt.coordinateMatrixToBinaryBlock(new JavaSparkContext(sc), coordinateMatrix, mc, true)
	 * </code></pre>
	 * 
	 * @param sc
	 * @param input
	 * @param mcIn
	 * @param outputEmptyBlocks
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> coordinateMatrixToBinaryBlock(JavaSparkContext sc,
			CoordinateMatrix input, MatrixCharacteristics mcIn, boolean outputEmptyBlocks) throws DMLRuntimeException 
	{
		//convert matrix entry rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = input.entries().toJavaRDD()
				.mapPartitionsToPair(new MatrixEntryToBinaryBlockFunction(mcIn));
		
		//inject empty blocks (if necessary) 
		if( outputEmptyBlocks && mcIn.mightHaveEmptyBlocks() ) {
			out = out.union( 
				SparkUtils.getEmptyBlockRDD(sc, mcIn) );
		}
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> coordinateMatrixToBinaryBlock(SparkContext sc,
			CoordinateMatrix input, MatrixCharacteristics mcIn, boolean outputEmptyBlocks) throws DMLRuntimeException 
	{
		return coordinateMatrixToBinaryBlock(new JavaSparkContext(sc), input, mcIn, true);
	}
	
	// Useful for printing, testing binary blocked RDD and also for external use.
	public static JavaRDD<String> binaryBlockToStringRDD(JavaPairRDD<MatrixIndexes, MatrixBlock> input, MatrixCharacteristics mcIn, String format) throws DMLRuntimeException {
		if(format.equals("text")) {
			JavaRDD<String> ijv = input.flatMap(new ConvertMatrixBlockToIJVLines(mcIn.getRowsPerBlock(), mcIn.getColsPerBlock()));
			return ijv;
		}
		else {
			throw new DMLRuntimeException("The output format:" + format + " is not implemented yet.");
		}
	}



	public static DataFrame stringDataFrameToVectorDataFrame(SQLContext sqlContext, DataFrame inputDF)
			throws DMLRuntimeException {

		StructField[] oldSchema = inputDF.schema().fields();
		//create the new schema
		StructField[] newSchema = new StructField[oldSchema.length];
		for(int i = 0; i < oldSchema.length; i++) {
			String colName = oldSchema[i].name();
			newSchema[i] = DataTypes.createStructField(colName, new VectorUDT(), true);
		}

		//converter
		class StringToVector implements Function<Tuple2<Row, Long>, Row> {
			private static final long serialVersionUID = -4733816995375745659L;
			@Override
			public Row call(Tuple2<Row, Long> arg0) throws Exception {
				Row oldRow = arg0._1;
				int oldNumCols = oldRow.length();
				if (oldNumCols > 1) {
					throw new DMLRuntimeException("The row must have at most one column");
				}

				// parse the various strings. i.e
				// ((1.2,4.3, 3.4))  or (1.2, 3.4, 2.2) or (1.2 3.4)
				// [[1.2,34.3, 1.2, 1.2]] or [1.2, 3.4] or [1.3 1.2]
				Object [] fields = new Object[oldNumCols];
				ArrayList<Object> fieldsArr = new ArrayList<Object>();
				for (int i = 0; i < oldRow.length(); i++) {
					Object ci=oldRow.get(i);
					if (ci instanceof String) {
						String cis = (String)ci;
						StringBuffer sb = new StringBuffer(cis.trim());
						for (int nid=0; i < 2; i++) { //remove two level nesting
							if ((sb.charAt(0) == '(' && sb.charAt(sb.length() - 1) == ')') ||
									(sb.charAt(0) == '[' && sb.charAt(sb.length() - 1) == ']')
									) {
								sb.deleteCharAt(0);
								sb.setLength(sb.length() - 1);
							}
						}
						//have the replace code
						String ncis = "[" + sb.toString().replaceAll(" *, *", ",") + "]";
						Vector v = Vectors.parse(ncis);
						fieldsArr.add(v);
					} else {
						throw new DMLRuntimeException("Only String is supported");
					}
				}
				Row row = RowFactory.create(fieldsArr.toArray());
				return row;
			}
		}

		//output DF
		JavaRDD<Row> newRows = inputDF.rdd().toJavaRDD().zipWithIndex().map(new StringToVector());
		// DataFrame outDF = sqlContext.createDataFrame(newRows, new StructType(newSchema)); //TODO investigate why it doesn't work
		DataFrame outDF = sqlContext.createDataFrame(newRows.rdd(),
				DataTypes.createStructType(newSchema));

		return outDF;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> vectorDataFrameToBinaryBlock(SparkContext sc,
			DataFrame inputDF, MatrixCharacteristics mcOut, boolean containsID, String vectorColumnName) throws DMLRuntimeException {
		return vectorDataFrameToBinaryBlock(new JavaSparkContext(sc), inputDF, mcOut, containsID, vectorColumnName);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> vectorDataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame inputDF, MatrixCharacteristics mcOut, boolean containsID, String vectorColumnName)
			throws DMLRuntimeException {
		
		if(containsID) {
			inputDF = dropColumn(inputDF.sort("__INDEX"), "__INDEX");
		}
		
		DataFrame df = inputDF.select(vectorColumnName);
			
		//determine unknown dimensions and sparsity if required
		if( !mcOut.dimsKnown(true) ) {
			Accumulator<Double> aNnz = sc.accumulator(0L);
			JavaRDD<Row> tmp = df.javaRDD().map(new DataFrameAnalysisFunction(aNnz, true));
			long rlen = tmp.count();
			long clen = ((Vector) tmp.first().get(0)).size();
			long nnz = UtilFunctions.toLong(aNnz.value());
			mcOut.set(rlen, clen, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), nnz);
		}
		
		JavaPairRDD<Row, Long> prepinput = df.javaRDD()
				.zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(
					new DataFrameToBinaryBlockFunction(mcOut, true));
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}
	
	/**
	 * Adding utility to support for dropping columns for older Spark versions.
	 * @param df
	 * @param column
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DataFrame dropColumn(DataFrame df, String column) throws DMLRuntimeException {
		ArrayList<String> columnToSelect = new ArrayList<String>();
		String firstCol = null;
		boolean colPresent = false;
		for(String col : df.columns()) {
			if(col.equals(column)) {
				colPresent = true;
			}
			else if(firstCol == null) {
				firstCol = col;
			}
			else {
				columnToSelect.add(col);
			}
		}
		
		if(!colPresent) {
			throw new DMLRuntimeException("The column \"" + column + "\" is not present in the dataframe.");
		}
		else if(firstCol == null) {
			throw new DMLRuntimeException("No column other than \"" + column + "\" present in the dataframe.");
		}
		
		// Round about way to do in Java (not exposed in Spark 1.3.0): df = df.drop("__INDEX");
		return df.select(firstCol, scala.collection.JavaConversions.asScalaBuffer(columnToSelect).toList());
	}
	
	public static DataFrame projectColumns(DataFrame df, ArrayList<String> columns) throws DMLRuntimeException {
		ArrayList<String> columnToSelect = new ArrayList<String>();
		for(int i = 1; i < columns.size(); i++) {
			columnToSelect.add(columns.get(i));
		}
		return df.select(columns.get(0), scala.collection.JavaConversions.asScalaBuffer(columnToSelect).toList());
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(SparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID) throws DMLRuntimeException {
		return dataFrameToBinaryBlock(new JavaSparkContext(sc), df, mcOut, containsID, null);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(SparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, String [] columns) throws DMLRuntimeException {
		ArrayList<String> columns1 = new ArrayList<String>(Arrays.asList(columns));
		return dataFrameToBinaryBlock(new JavaSparkContext(sc), df, mcOut, false, columns1);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(SparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, ArrayList<String> columns) throws DMLRuntimeException {
		return dataFrameToBinaryBlock(new JavaSparkContext(sc), df, mcOut, false, columns);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(SparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID, String [] columns) 
			throws DMLRuntimeException {
		ArrayList<String> columns1 = new ArrayList<String>(Arrays.asList(columns));
		return dataFrameToBinaryBlock(new JavaSparkContext(sc), df, mcOut, containsID, columns1);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(SparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID, ArrayList<String> columns) 
			throws DMLRuntimeException {
		return dataFrameToBinaryBlock(new JavaSparkContext(sc), df, mcOut, containsID, columns);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID) throws DMLRuntimeException {
		return dataFrameToBinaryBlock(sc, df, mcOut, containsID, null);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, ArrayList<String> columns) throws DMLRuntimeException {
		return dataFrameToBinaryBlock(sc, df, mcOut, false, columns);
	}
	
	public static MatrixBlock convertPy4JArrayToMB(byte [] data, int rlen, int clen) throws DMLRuntimeException {
		return convertPy4JArrayToMB(data, rlen, clen, false);
	}
	
	public static MatrixBlock convertSciPyCOOToMB(byte [] data, byte [] row, byte [] col, int rlen, int clen, int nnz) throws DMLRuntimeException {
		MatrixBlock mb = new MatrixBlock(rlen, clen, true);
		mb.allocateSparseRowsBlock(false);
		ByteBuffer buf1 = ByteBuffer.wrap(data);
		buf1.order(ByteOrder.nativeOrder());
		ByteBuffer buf2 = ByteBuffer.wrap(row);
		buf2.order(ByteOrder.nativeOrder());
		ByteBuffer buf3 = ByteBuffer.wrap(col);
		buf3.order(ByteOrder.nativeOrder());
		for(int i = 0; i < nnz; i++) {
			double val = buf1.getDouble();
			int rowIndex = buf2.getInt();
			int colIndex = buf3.getInt();
			mb.setValue(rowIndex, colIndex, val); // TODO: Improve the performance
		}
		return mb;
	}
	
	public static MatrixBlock convertPy4JArrayToMB(byte [] data, int rlen, int clen, boolean isSparse) throws DMLRuntimeException {
		MatrixBlock mb = new MatrixBlock(rlen, clen, isSparse, -1);
		if(isSparse) {
			throw new DMLRuntimeException("Convertion to sparse format not supported");
		}
		else {
			double [] denseBlock = new double[rlen*clen];
			ByteBuffer buf = ByteBuffer.wrap(data);
			buf.order(ByteOrder.nativeOrder());
			for(int i = 0; i < rlen*clen; i++) {
				denseBlock[i] = buf.getDouble();
			}
			mb.init( denseBlock, rlen, clen );
		}
		mb.examSparsity();
		return mb;
	}
	
	public static byte [] convertMBtoPy4JDenseArr(MatrixBlock mb) throws DMLRuntimeException {
		byte [] ret = null;
		if(mb.isInSparseFormat()) {
			throw new DMLRuntimeException("Sparse to dense conversion is not yet implemented");
		}
		else {
			double [] denseBlock = mb.getDenseBlock();
			if(denseBlock == null) {
				throw new DMLRuntimeException("Sparse to dense conversion is not yet implemented");
			}
			int times = Double.SIZE / Byte.SIZE;
			ret = new byte[denseBlock.length * times];
			for(int i=0;i < denseBlock.length;i++){
		        ByteBuffer.wrap(ret, i*times, times).order(ByteOrder.nativeOrder()).putDouble(denseBlock[i]);
			}
		}
		return ret;
	}
	
	/**
	 * Converts DataFrame into binary blocked RDD. 
	 * Note: mcOut will be set if you don't know the dimensions.
	 * @param sc
	 * @param df
	 * @param mcOut
	 * @param containsID
	 * @param columns
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlock(JavaSparkContext sc,
			DataFrame df, MatrixCharacteristics mcOut, boolean containsID, ArrayList<String> columns) 
			throws DMLRuntimeException {
		if(columns != null) {
			df = projectColumns(df, columns);
		}
		
		if(containsID) {
			df = dropColumn(df.sort("__INDEX"), "__INDEX");
		}
			
		//determine unknown dimensions and sparsity if required
		if( !mcOut.dimsKnown(true) ) {
			Accumulator<Double> aNnz = sc.accumulator(0L);
			JavaRDD<Row> tmp = df.javaRDD().map(new DataFrameAnalysisFunction(aNnz, false));
			long rlen = tmp.count();
			long clen = containsID ? (df.columns().length - 1) : df.columns().length;
			long nnz = UtilFunctions.toLong(aNnz.value());
			mcOut.set(rlen, clen, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), nnz);
		}
		
		JavaPairRDD<Row, Long> prepinput = df.javaRDD()
				.zipWithIndex(); //zip row index
		
		//convert csv rdd to binary block rdd (w/ partial blocks)
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
				prepinput.mapPartitionsToPair(
					new DataFrameToBinaryBlockFunction(mcOut, false));
		
		//aggregate partial matrix blocks
		out = RDDAggregateUtils.mergeByKey( out ); 
		
		return out;
	}
	
	public static DataFrame binaryBlockToVectorDataFrame(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockRDD, 
			MatrixCharacteristics mc, SQLContext sqlContext) throws DMLRuntimeException {
		long rlen = mc.getRows(); long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock(); int bclen = mc.getColsPerBlock();
		// Very expensive operation here: groupByKey (where number of keys might be too large)
		JavaRDD<Row> rowsRDD = binaryBlockRDD.flatMapToPair(new ProjectRows(rlen, clen, brlen, bclen))
				.groupByKey().map(new ConvertDoubleArrayToRows(clen, bclen, true));
		
		int numColumns = (int) clen;
		if(numColumns <= 0) {
			throw new DMLRuntimeException("Output dimensions unknown after executing the script and hence cannot create the dataframe");
		}
		
		List<StructField> fields = new ArrayList<StructField>();
		// LongTypes throw an error: java.lang.Double incompatible with java.lang.Long
		fields.add(DataTypes.createStructField("__INDEX", DataTypes.DoubleType, false));
		fields.add(DataTypes.createStructField("C1", new VectorUDT(), false));
		// fields.add(DataTypes.createStructField("C1", DataTypes.createArrayType(DataTypes.DoubleType), false));
		
		// This will cause infinite recursion due to bug in Spark
		// https://issues.apache.org/jira/browse/SPARK-6999
		// return sqlContext.createDataFrame(rowsRDD, colNames); // where ArrayList<String> colNames
		return sqlContext.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
	}
	
	public static class AddRowID implements Function<Tuple2<Row,Long>, Row> {
		private static final long serialVersionUID = -3733816995375745659L;

		@Override
		public Row call(Tuple2<Row, Long> arg0) throws Exception {
			int oldNumCols = arg0._1.length();
			Object [] fields = new Object[oldNumCols + 1];
			for(int i = 0; i < oldNumCols; i++) {
				fields[i] = arg0._1.get(i);
			}
			fields[oldNumCols] = new Double(arg0._2 + 1);
			return RowFactory.create(fields);
		}
		
	}
	
	/**
	 * Add element indices as new column to DataFrame
	 * Note: Element indices start from 1
	 * @param df input data frame
	 * @param nameOfCol name of index column
	 * @return new data frame
	 */
	public static DataFrame addIDToDataFrame(DataFrame df, SQLContext sqlContext, String nameOfCol) {
		StructField[] oldSchema = df.schema().fields();
		StructField[] newSchema = new StructField[oldSchema.length + 1];
		for(int i = 0; i < oldSchema.length; i++) {
			newSchema[i] = oldSchema[i];
		}
		newSchema[oldSchema.length] = DataTypes.createStructField(nameOfCol, DataTypes.DoubleType, false);
		// JavaRDD<Row> newRows = df.rdd().toJavaRDD().map(new AddRowID());
		JavaRDD<Row> newRows = df.rdd().toJavaRDD().zipWithIndex().map(new AddRowID());
		return sqlContext.createDataFrame(newRows, new StructType(newSchema));
	}
	
	public static DataFrame binaryBlockToDataFrame(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockRDD, 
			MatrixCharacteristics mc, SQLContext sqlContext) throws DMLRuntimeException {
		long rlen = mc.getRows(); long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock(); int bclen = mc.getColsPerBlock();
		
		// Very expensive operation here: groupByKey (where number of keys might be too large)
		JavaRDD<Row> rowsRDD = binaryBlockRDD.flatMapToPair(new ProjectRows(rlen, clen, brlen, bclen))
				.groupByKey().map(new ConvertDoubleArrayToRows(clen, bclen, false));
		
		int numColumns = (int) clen;
		if(numColumns <= 0) {
			// numColumns = rowsRDD.first().length() - 1; // Ugly, so instead prefer to throw
			throw new DMLRuntimeException("Output dimensions unknown after executing the script and hence cannot create the dataframe");
		}
		
		List<StructField> fields = new ArrayList<StructField>();
		// LongTypes throw an error: java.lang.Double incompatible with java.lang.Long
		fields.add(DataTypes.createStructField("__INDEX", DataTypes.DoubleType, false)); 
		for(int i = 1; i <= numColumns; i++) {
			fields.add(DataTypes.createStructField("C" + i, DataTypes.DoubleType, false));
		}
		
		// This will cause infinite recursion due to bug in Spark
		// https://issues.apache.org/jira/browse/SPARK-6999
		// return sqlContext.createDataFrame(rowsRDD, colNames); // where ArrayList<String> colNames
		return sqlContext.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
	}
	
	private static class MatrixEntryToBinaryBlockFunction implements PairFlatMapFunction<Iterator<MatrixEntry>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 4907483236186747224L;
		
		private IJVToBinaryBlockFunctionHelper helper = null;
		public MatrixEntryToBinaryBlockFunction(MatrixCharacteristics mc) throws DMLRuntimeException {
			helper = new IJVToBinaryBlockFunctionHelper(mc);
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<MatrixEntry> arg0) throws Exception {
			return helper.convertToBinaryBlock(arg0, RDDConverterTypes.MATRIXENTRY_TO_MATRIXCELL);
		}

	}

	public static class DataFrameAnalysisFunction implements Function<Row,Row>  {
		private static final long serialVersionUID = 5705371332119770215L;
		private RowAnalysisFunctionHelper helper = null;
		boolean isVectorBasedRDD;
		public DataFrameAnalysisFunction( Accumulator<Double> aNnz, boolean isVectorBasedRDD) {
			helper = new RowAnalysisFunctionHelper(aNnz);
			this.isVectorBasedRDD = isVectorBasedRDD;
		}

		@Override
		public Row call(Row arg0) throws Exception {
			if(isVectorBasedRDD)
				return helper.analyzeVector(arg0);
			else
				return helper.analyzeRow(arg0);
		}
		
	}
	
	private static class CSVToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Text,Long>>,MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = 1501589201971233542L;
		
		private RowToBinaryBlockFunctionHelper helper = null; 
		
		public CSVToBinaryBlockFunction(MatrixCharacteristics mc, String delim, boolean fill, double fillValue) {
			helper = new RowToBinaryBlockFunctionHelper(mc, delim, fill, fillValue);
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Text, Long>> arg0) throws Exception {
			return helper.convertToBinaryBlock(arg0, RDDConverterTypes.TEXT_TO_DOUBLEARR);
		}
		
	}
	
	public static class DataFrameToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Row,Long>>,MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = 653447740362447236L;
		private RowToBinaryBlockFunctionHelper helper = null; 
		boolean isVectorBasedDF;
		
		public DataFrameToBinaryBlockFunction(MatrixCharacteristics mc, boolean isVectorBasedDF) {
			helper = new RowToBinaryBlockFunctionHelper(mc);
			this.isVectorBasedDF = isVectorBasedDF;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Row, Long>> arg0) throws Exception {
			if(isVectorBasedDF)
				return helper.convertToBinaryBlock(arg0, RDDConverterTypes.VECTOR_TO_DOUBLEARR);
			else
				return helper.convertToBinaryBlock(arg0, RDDConverterTypes.ROW_TO_DOUBLEARR);
		}
		
	}
	
	private static class RowAnalysisFunctionHelper implements Serializable 
	{
		private static final long serialVersionUID = 2310303223289674477L;

		private Accumulator<Double> _aNnz = null;
		private String _delim = null;
		
		public RowAnalysisFunctionHelper( Accumulator<Double> aNnz ) {
			_aNnz = aNnz;
		}
		
		public RowAnalysisFunctionHelper( Accumulator<Double> aNnz, String delim ) {
			_aNnz = aNnz;
			_delim = delim;
		}
		
		public String analyzeText(Text v1) throws Exception {
			//parse input line
			String line = v1.toString();
			String[] cols = IOUtilFunctions.split(line, _delim);
			
			//determine number of non-zeros of row (w/o string parsing)
			long lnnz = 0;
			for( String col : cols ) {
				if( !col.isEmpty() && !col.equals("0") && !col.equals("0.0") ) {
					lnnz++;
				}
			}
			
			//update counters
			_aNnz.add( (double)lnnz );
			
			return line;
		}
		
		public Row analyzeRow(Row arg0) throws Exception {
			//determine number of non-zeros of row
			long lnnz = 0;
			if(arg0 != null) {
				for(int i = 0; i < arg0.length(); i++) {
					if(RowToBinaryBlockFunctionHelper.getDoubleValue(arg0, i) != 0) {
						lnnz++;
					}
				}
			}
			else {
				throw new Exception("Error while analyzing row");
			}
			
			//update counters
			_aNnz.add( (double)lnnz );
			
			return arg0;
		}
		
		public Row analyzeVector(Row row)  {
			Vector vec = (Vector) row.get(0); // assumption: 1 column DF
			long lnnz = 0;
			for(int i = 0; i < vec.size(); i++) {
				if(vec.apply(i) != 0) { 
					lnnz++;
				}
			}
			
			//update counters
			_aNnz.add( (double)lnnz );
			return row;
		}
	}
	
	private static class IJVToBinaryBlockFunctionHelper implements Serializable {
		private static final long serialVersionUID = -7952801318564745821L;
		//internal buffer size (aligned w/ default matrix block size)
		private static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
		private int _bufflen = -1;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public IJVToBinaryBlockFunctionHelper(MatrixCharacteristics mc) throws DMLRuntimeException
		{
			if(!mc.dimsKnown()) {
				throw new DMLRuntimeException("The dimensions need to be known in given MatrixCharacteristics for given input RDD");
			}
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			
			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen*_clen, BUFFER_SIZE);
			
		}
		
		// ----------------------------------------------------
		// Can extend this by having type hierarchy
		public Tuple2<MatrixIndexes, MatrixCell> textToMatrixCell(Text txt) {
			FastStringTokenizer st = new FastStringTokenizer(' ');
			//get input string (ignore matrix market comments)
			String strVal = txt.toString();
			if( strVal.startsWith("%") ) 
				return null;
			
			//parse input ijv triple
			st.reset( strVal );
			long row = st.nextLong();
			long col = st.nextLong();
			double val = st.nextDouble();
			MatrixIndexes indx = new MatrixIndexes(row, col);
			MatrixCell cell = new MatrixCell(val);
			return new Tuple2<MatrixIndexes, MatrixCell>(indx, cell);
		}
		
		public Tuple2<MatrixIndexes, MatrixCell> matrixEntryToMatrixCell(MatrixEntry entry) {
			MatrixIndexes indx = new MatrixIndexes(entry.i(), entry.j());
			MatrixCell cell = new MatrixCell(entry.value());
			return new Tuple2<MatrixIndexes, MatrixCell>(indx, cell);
		}
		
		// ----------------------------------------------------
		
		Iterable<Tuple2<MatrixIndexes, MatrixBlock>> convertToBinaryBlock(Object arg0, RDDConverterTypes converter)  throws Exception {
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);
		
			Iterator<?> iter = (Iterator<?>) arg0;
			while( iter.hasNext() ) {
				Tuple2<MatrixIndexes, MatrixCell> cell = null;
				switch(converter) {
					case MATRIXENTRY_TO_MATRIXCELL:
						cell = matrixEntryToMatrixCell((MatrixEntry) iter.next());
						break;
						
					case TEXT_TO_MATRIX_CELL:
						cell = textToMatrixCell((Text) iter.next());
						break;
					
					default:
						throw new Exception("Invalid converter for IJV data:" + converter.toString());
				}
				
				if(cell == null) {
					continue;
				}
				
				//flush buffer if necessary
				if( rbuff.getSize() >= rbuff.getCapacity() )
					flushBufferToList(rbuff, ret);
				
				//add value to reblock buffer
				rbuff.appendCell(cell._1.getRowIndex(), cell._1.getColumnIndex(), cell._2.getValue());
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
	
	/**
	 * This functions allows to map rdd partitions of csv rows into a set of partial binary blocks.
	 * 
	 * NOTE: For this csv to binary block function, we need to hold all output blocks per partition 
	 * in-memory. Hence, we keep state of all column blocks and aggregate row segments into these blocks. 
	 * In terms of memory consumption this is better than creating partial blocks of row segments.
	 * 
	 */
	private static class RowToBinaryBlockFunctionHelper implements Serializable 
	{
		private static final long serialVersionUID = -4948430402942717043L;
		
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		private String _delim = null;
		private boolean _fill = false;
		private double _fillValue = 0;
		
		public RowToBinaryBlockFunctionHelper(MatrixCharacteristics mc)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
		}
		
		public RowToBinaryBlockFunctionHelper(MatrixCharacteristics mc, String delim, boolean fill, double fillValue)
		{
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_delim = delim;
			_fill = fill;
			_fillValue = fillValue;
		}
		
		boolean emptyFound = false;
		
		// ----------------------------------------------------
		public double[] textToDoubleArray(Text row) {
			String[] parts = IOUtilFunctions.split(row.toString(), _delim);
			double[] ret = new double[parts.length];
			int ix = 0;
			for(String part : parts) {
				emptyFound |= part.isEmpty() && !_fill;
				double val = (part.isEmpty() && _fill) ?
						_fillValue : Double.parseDouble(part);
				ret[ix++] = val;
			}
			return ret;
		}
		public double[] rowToDoubleArray(Row row) throws Exception {
			double[] ret = new double[row.length()];
			for(int i = 0; i < row.length(); i++) {
				ret[i] = getDoubleValue(row, i);
			}
			return ret;
		}
		
		public double[] vectorToDoubleArray(Vector arg) throws Exception {
			return arg.toDense().values();
		}
		// ----------------------------------------------------

		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> convertToBinaryBlock(Object arg0, RDDConverterTypes converter) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();

			int ncblks = (int)Math.ceil((double)_clen/_bclen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			
			@SuppressWarnings("unchecked")
			Iterator<Tuple2<?,Long>> iter = (Iterator<Tuple2<?, Long>>) arg0;
			while( iter.hasNext() )
			{
				Tuple2<?,Long> tmp = iter.next();
				// String row = tmp._1();
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
				emptyFound = false;
				double[] parts = null;
				switch(converter) {
					case TEXT_TO_DOUBLEARR:
						parts = textToDoubleArray((Text) tmp._1());
						break;
					case ROW_TO_DOUBLEARR:
						parts = rowToDoubleArray((Row) tmp._1());
						break;
					case VECTOR_TO_DOUBLEARR:
						parts = vectorToDoubleArray((Vector) ((Row) tmp._1()).get(0));
						break;
					default:
						throw new Exception("Invalid converter for row-based data:" + converter.toString());
				}
				
				for( int cix=1, pix=0; cix<=ncblks; cix++ ) 
				{
					int lclen = (int)UtilFunctions.computeBlockSize(_clen, cix, _bclen);				
					for( int j=0; j<lclen; j++ ) {
						double val = parts[pix++];
						mb[cix-1].appendValue(pos, j, val);
					}	
				}
		
				//sanity check empty cells filled w/ values
				if(converter == RDDConverterTypes.TEXT_TO_DOUBLEARR)
					IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(((Text) tmp._1()).toString(), _fill, emptyFound);
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
				mb[cix-1] = new MatrixBlock(lrlen, lclen, false);		
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
		
		public static double getDoubleValue(Row row, int index) throws Exception {
			try {
				return row.getDouble(index);
			} catch(Exception e) {
				try {
					// Causes lock-contention for Java 7
					return Double.parseDouble(row.get(index).toString());
				}
				catch(Exception e1) {
					throw new Exception("Only double types are supported as input to SystemML. The input argument is \'" + row.get(index) + "\'");
				}
			}
		}
	}
}
