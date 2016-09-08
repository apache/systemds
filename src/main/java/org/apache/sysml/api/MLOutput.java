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

package org.apache.sysml.api;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.functions.GetMLBlock;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.Tuple2;

/**
 * This is a simple container object that returns the output of execute from MLContext 
 *
 */
public class MLOutput {
	
	Map<String, JavaPairRDD<?,?>> _outputs;
	private Map<String, MatrixCharacteristics> _outMetadata = null;
	
	public MatrixBlock getMatrixBlock(String varName) throws DMLRuntimeException {
		MatrixCharacteristics mc = getMatrixCharacteristics(varName);
		// The matrix block is always pushed to an RDD and then we do collect
		// We can later avoid this by returning symbol table rather than "Map<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> _outputs"
		MatrixBlock mb = SparkExecutionContext.toMatrixBlock(getBinaryBlockedRDD(varName), (int) mc.getRows(), (int) mc.getCols(), 
				mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros());
		return mb;
	}

	public MLOutput(Map<String, JavaPairRDD<?,?>> outputs, Map<String, MatrixCharacteristics> outMetadata) {
		this._outputs = outputs;
		this._outMetadata = outMetadata;
	}
	
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryBlockedRDD(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return (JavaPairRDD<MatrixIndexes,MatrixBlock>) _outputs.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the outputs.");
	}
	
	@SuppressWarnings("unchecked")
	public JavaPairRDD<Long,FrameBlock> getFrameBinaryBlockedRDD(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return (JavaPairRDD<Long,FrameBlock>)_outputs.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the outputs.");
	}
	
	public MatrixCharacteristics getMatrixCharacteristics(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return _outMetadata.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
	
	/**
	 * Note, the output DataFrame has an additional column ID.
	 * An easy way to get DataFrame without ID is by df.sort("__INDEX").drop("__INDEX")
	 * @param sqlContext
	 * @param varName
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
		if(rdd != null) {
			MatrixCharacteristics mc = _outMetadata.get(varName);
			return RDDConverterUtilsExt.binaryBlockToDataFrame(rdd, mc, sqlContext);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
	
	/**
	 * 
	 * @param sqlContext
	 * @param varName
	 * @param outputVector if true, returns DataFrame with two column: ID and org.apache.spark.mllib.linalg.Vector
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName, boolean outputVector) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		if(outputVector) {
			JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
			if(rdd != null) {
				MatrixCharacteristics mc = _outMetadata.get(varName);
				return RDDConverterUtilsExt.binaryBlockToVectorDataFrame(rdd, mc, sqlContext);
			}
			throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
		}
		else {
			return getDF(sqlContext, varName);
		}
		
	}
	
	/**
	 * This methods improves the performance of MLPipeline wrappers.
	 * @param sqlContext
	 * @param varName
	 * @param range range is inclusive
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName, Map<String, Tuple2<Long, Long>> range) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		JavaPairRDD<MatrixIndexes,MatrixBlock> binaryBlockRDD = getBinaryBlockedRDD(varName);
		if(binaryBlockRDD == null) {
			throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
		}
		MatrixCharacteristics mc = _outMetadata.get(varName);
		long rlen = mc.getRows(); long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock(); int bclen = mc.getColsPerBlock();
		
		ArrayList<Tuple2<String, Tuple2<Long, Long>>> alRange = new ArrayList<Tuple2<String, Tuple2<Long, Long>>>();
		for(Entry<String, Tuple2<Long, Long>> e : range.entrySet()) {
			alRange.add(new Tuple2<String, Tuple2<Long,Long>>(e.getKey(), e.getValue()));
		}
		
		// Very expensive operation here: groupByKey (where number of keys might be too large)
		JavaRDD<Row> rowsRDD = binaryBlockRDD.flatMapToPair(new ProjectRows(rlen, clen, brlen, bclen))
				.groupByKey().map(new ConvertDoubleArrayToRangeRows(clen, bclen, alRange));

		int numColumns = (int) clen;
		if(numColumns <= 0) {
			throw new DMLRuntimeException("Output dimensions unknown after executing the script and hence cannot create the dataframe");
		}
		
		List<StructField> fields = new ArrayList<StructField>();
		// LongTypes throw an error: java.lang.Double incompatible with java.lang.Long
		fields.add(DataTypes.createStructField("__INDEX", DataTypes.DoubleType, false));
		for(int k = 0; k < alRange.size(); k++) {
			String colName = alRange.get(k)._1;
			long low = alRange.get(k)._2._1;
			long high = alRange.get(k)._2._2;
			if(low != high)
				fields.add(DataTypes.createStructField(colName, new VectorUDT(), false));
			else
				fields.add(DataTypes.createStructField(colName, DataTypes.DoubleType, false));
		}
		
		// This will cause infinite recursion due to bug in Spark
		// https://issues.apache.org/jira/browse/SPARK-6999
		// return sqlContext.createDataFrame(rowsRDD, colNames); // where ArrayList<String> colNames
		return sqlContext.createDataFrame(rowsRDD.rdd(), DataTypes.createStructType(fields));
		
	}
	
	public JavaRDD<String> getStringRDD(String varName, String format) throws DMLRuntimeException {
		if(format.equals("text")) {
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryRDD = getBinaryBlockedRDD(varName);
			MatrixCharacteristics mcIn = getMatrixCharacteristics(varName); 
			return RDDConverterUtilsExt.binaryBlockToStringRDD(binaryRDD, mcIn, format);
		}
		else {
			throw new DMLRuntimeException("The output format:" + format + " is not implemented yet.");
		}
		
	}
	
	public JavaRDD<String> getStringFrameRDD(String varName, String format, CSVFileFormatProperties fprop ) throws DMLRuntimeException {
		JavaPairRDD<Long, FrameBlock> binaryRDD = getFrameBinaryBlockedRDD(varName);
		MatrixCharacteristics mcIn = getMatrixCharacteristics(varName); 
		if(format.equals("csv")) {
			return FrameRDDConverterUtils.binaryBlockToCsv(binaryRDD, mcIn, fprop, false);
		}
		else if(format.equals("text")) {
			return FrameRDDConverterUtils.binaryBlockToTextCell(binaryRDD, mcIn);
		}
		else {
			throw new DMLRuntimeException("The output format:" + format + " is not implemented yet.");
		}
		
	}
	
	public DataFrame getDataFrameRDD(String varName, JavaSparkContext jsc) throws DMLRuntimeException {
		JavaPairRDD<Long, FrameBlock> binaryRDD = getFrameBinaryBlockedRDD(varName);
		MatrixCharacteristics mcIn = getMatrixCharacteristics(varName);
		return FrameRDDConverterUtils.binaryBlockToDataFrame(binaryRDD, mcIn, jsc);
	}
	
	public MLMatrix getMLMatrix(MLContext ml, SQLContext sqlContext, String varName) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		else if(ml == null) {
			throw new DMLRuntimeException("MLContext is not created.");
		}
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
		if(rdd != null) {
			MatrixCharacteristics mc = getMatrixCharacteristics(varName);
			StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
			return new MLMatrix(sqlContext.createDataFrame(rdd.map(new GetMLBlock()).rdd(), schema), mc, ml);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
	
//	/**
//	 * Experimental: Please use this with caution as it will fail in many corner cases.
//	 * @return org.apache.spark.mllib.linalg.distributed.BlockMatrix
//	 * @throws DMLRuntimeException 
//	 */
//	public BlockMatrix getMLLibBlockedMatrix(MLContext ml, SQLContext sqlContext, String varName) throws DMLRuntimeException {
//		return getMLMatrix(ml, sqlContext, varName).toBlockedMatrix();
//	}
	
	public static class ProjectRows implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, Long, Tuple2<Long, Double[]>> {
		private static final long serialVersionUID = -4792573268900472749L;
		long rlen; long clen;
		int brlen; int bclen;
		public ProjectRows(long rlen, long clen, int brlen, int bclen) {
			this.rlen = rlen;
			this.clen = clen;
			this.brlen = brlen;
			this.bclen = bclen;
		}

		@Override
		public Iterable<Tuple2<Long, Tuple2<Long, Double[]>>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			// ------------------------------------------------------------------
    		//	Compute local block size: 
    		// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
    		// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
    		long blockRowIndex = kv._1.getRowIndex();
    		long blockColIndex = kv._1.getColumnIndex();
    		int lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
    		int lclen = UtilFunctions.computeBlockSize(clen, blockColIndex, bclen);
    		// ------------------------------------------------------------------
			
			long startRowIndex = (kv._1.getRowIndex()-1) * bclen + 1;
			MatrixBlock blk = kv._2;
			ArrayList<Tuple2<Long, Tuple2<Long, Double[]>>> retVal = new ArrayList<Tuple2<Long,Tuple2<Long,Double[]>>>();
			for(int i = 0; i < lrlen; i++) {
				Double[] partialRow = new Double[lclen];
				for(int j = 0; j < lclen; j++) {
					partialRow[j] = blk.getValue(i, j);
				}
				retVal.add(new Tuple2<Long, Tuple2<Long,Double[]>>(startRowIndex + i, new Tuple2<Long,Double[]>(kv._1.getColumnIndex(), partialRow)));
			}
			return retVal;
		}
	}
	
	public static class ConvertDoubleArrayToRows implements Function<Tuple2<Long, Iterable<Tuple2<Long, Double[]>>>, Row> {
		private static final long serialVersionUID = 4441184411670316972L;
		
		int bclen; long clen;
		boolean outputVector;
		public ConvertDoubleArrayToRows(long clen, int bclen, boolean outputVector) {
			this.bclen = bclen;
			this.clen = clen;
			this.outputVector = outputVector;
		}

		@Override
		public Row call(Tuple2<Long, Iterable<Tuple2<Long, Double[]>>> arg0)
				throws Exception {
			
			HashMap<Long, Double[]> partialRows = new HashMap<Long, Double[]>();
			int sizeOfPartialRows = 0;
			for(Tuple2<Long, Double[]> kv : arg0._2) {
				partialRows.put(kv._1, kv._2);
				sizeOfPartialRows += kv._2.length;
			}
			
			// Insert first row as row index
			Object[] row = null;
			if(outputVector) {
				row = new Object[2];
				double [] vecVals = new double[sizeOfPartialRows];
				
				for(long columnBlockIndex = 1; columnBlockIndex <= partialRows.size(); columnBlockIndex++) {
					if(partialRows.containsKey(columnBlockIndex)) {
						Double [] array = partialRows.get(columnBlockIndex);
						// ------------------------------------------------------------------
						//	Compute local block size: 
						int lclen = UtilFunctions.computeBlockSize(clen, columnBlockIndex, bclen);
						// ------------------------------------------------------------------
						if(array.length != lclen) {
							throw new Exception("Incorrect double array provided by ProjectRows");
						}
						for(int i = 0; i < lclen; i++) {
							vecVals[(int) ((columnBlockIndex-1)*bclen + i)] = array[i];
						}
					}
					else {
						throw new Exception("The block for column index " + columnBlockIndex + " is missing. Make sure the last instruction is not returning empty blocks");
					}
				}
				
				long rowIndex = arg0._1;
				row[0] = (double) rowIndex;
				row[1] = new DenseVector(vecVals); // breeze.util.JavaArrayOps.arrayDToDv(vecVals);
			}
			else {
				row = new Double[sizeOfPartialRows + 1];
				long rowIndex = arg0._1;
				row[0] = (double) rowIndex;
				for(long columnBlockIndex = 1; columnBlockIndex <= partialRows.size(); columnBlockIndex++) {
					if(partialRows.containsKey(columnBlockIndex)) {
						Double [] array = partialRows.get(columnBlockIndex);
						// ------------------------------------------------------------------
						//	Compute local block size: 
						int lclen = UtilFunctions.computeBlockSize(clen, columnBlockIndex, bclen);
						// ------------------------------------------------------------------
						if(array.length != lclen) {
							throw new Exception("Incorrect double array provided by ProjectRows");
						}
						for(int i = 0; i < lclen; i++) {
							row[(int) ((columnBlockIndex-1)*bclen + i) + 1] = array[i];
						}
					}
					else {
						throw new Exception("The block for column index " + columnBlockIndex + " is missing. Make sure the last instruction is not returning empty blocks");
					}
				}
			}
			Object[] row_fields = row;
			return RowFactory.create(row_fields);
		}
	}
	
	
	public static class ConvertDoubleArrayToRangeRows implements Function<Tuple2<Long, Iterable<Tuple2<Long, Double[]>>>, Row> {
		private static final long serialVersionUID = 4441184411670316972L;
		
		int bclen; long clen;
		ArrayList<Tuple2<String, Tuple2<Long, Long>>> range;
		public ConvertDoubleArrayToRangeRows(long clen, int bclen, ArrayList<Tuple2<String, Tuple2<Long, Long>>> range) {
			this.bclen = bclen;
			this.clen = clen;
			this.range = range;
		}

		@Override
		public Row call(Tuple2<Long, Iterable<Tuple2<Long, Double[]>>> arg0)
				throws Exception {
			
			HashMap<Long, Double[]> partialRows = new HashMap<Long, Double[]>();
			int sizeOfPartialRows = 0;
			for(Tuple2<Long, Double[]> kv : arg0._2) {
				partialRows.put(kv._1, kv._2);
				sizeOfPartialRows += kv._2.length;
			}
			
			// Insert first row as row index
			Object[] row = new Object[range.size() + 1];
			
			double [] vecVals = new double[sizeOfPartialRows];
			
			for(long columnBlockIndex = 1; columnBlockIndex <= partialRows.size(); columnBlockIndex++) {
				if(partialRows.containsKey(columnBlockIndex)) {
					Double [] array = partialRows.get(columnBlockIndex);
					// ------------------------------------------------------------------
					//	Compute local block size: 
					int lclen = UtilFunctions.computeBlockSize(clen, columnBlockIndex, bclen);
					// ------------------------------------------------------------------
					if(array.length != lclen) {
						throw new Exception("Incorrect double array provided by ProjectRows");
					}
					for(int i = 0; i < lclen; i++) {
						vecVals[(int) ((columnBlockIndex-1)*bclen + i)] = array[i];
					}
				}
				else {
					throw new Exception("The block for column index " + columnBlockIndex + " is missing. Make sure the last instruction is not returning empty blocks");
				}
			}
			
			long rowIndex = arg0._1;
			row[0] = (double) rowIndex;
			
			int i = 1;
			
			//for(Entry<String, Tuple2<Long, Long>> e : range.entrySet()) {
			for(int k = 0; k < range.size(); k++) {
				long low = range.get(k)._2._1;
				long high = range.get(k)._2._2;
				
				if(high < low) {
					throw new Exception("Incorrect range:" + high + "<" + low);
				}
				
				if(low == high) {
					row[i] = vecVals[(int) (low - 1)];
				}
				else {
					int lengthOfVector = (int) (high - low + 1);
					double [] tempVector = new double[lengthOfVector];
					for(int j = 0; j < lengthOfVector; j++) {
						tempVector[j] = vecVals[(int) (low + j - 1)];
					}
					row[i] = new DenseVector(tempVector);
				}
				
				i++;
			}

			return RowFactory.create(row);
		}
	}
}