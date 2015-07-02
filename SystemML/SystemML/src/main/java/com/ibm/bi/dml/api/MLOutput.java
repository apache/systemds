/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * This is a simple container object that returns the output of execute from MLContext 
 *
 */
public class MLOutput {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	
	HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> _outputs;
	private HashMap<String, MatrixCharacteristics> _outMetadata = null;
	
	public MLOutput(HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> outputs, HashMap<String, MatrixCharacteristics> outMetadata) {
		this._outputs = outputs;
		this._outMetadata = outMetadata;
	}
	
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryBlockedRDD(String varName) {
		if(_outputs.containsKey(varName)) {
			return _outputs.get(varName);
		}
		return null;
	}
	
	public DataFrame getDF(SQLContext sqlContext, String varName) {
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
		if(rdd != null) {
			return getDF(sqlContext, rdd, varName);
		}
		return null;
	}
	
	private DataFrame getDF(SQLContext sqlContext, JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockRDD, String varName) {
		long rlen = _outMetadata.get(varName).getRows(); long clen = _outMetadata.get(varName).getCols();
		int brlen = _outMetadata.get(varName).getRowsPerBlock(); int bclen = _outMetadata.get(varName).getColsPerBlock();
		
		// Very expensive operation here: groupByKey (where number of keys might be too large)
		JavaRDD<Row> rowsRDD = binaryBlockRDD.flatMapToPair(new ProjectRows(rlen, clen, brlen, bclen))
				.groupByKey().map(new ConvertDoubleArrayToRows(clen, bclen));
		
		int numColumns = (int) clen;
		if(numColumns <= 0) {
			numColumns = rowsRDD.first().length() - 1; // Ugly
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
			
			long startRowIndex = (kv._1.getRowIndex()-1) * bclen;
			MatrixBlock blk = kv._2;
			ArrayList<Tuple2<Long, Tuple2<Long, Double[]>>> retVal = new ArrayList<Tuple2<Long,Tuple2<Long,Double[]>>>();
			for(int i = 0; i < lrlen; i++) {
				Double[] partialRow = new Double[bclen];
				for(int j = 0; j < lclen; j++) {
					partialRow[j] = blk.getValue(i, j);
				}
				retVal.add(new Tuple2<Long, Tuple2<Long,Double[]>>(startRowIndex + i, new Tuple2<Long,Double[]>(kv._1.getColumnIndex(), partialRow)));
			}
			return (Iterable<Tuple2<Long, Tuple2<Long, Double[]>>>) retVal;
		}
	}
	
	public static class ConvertDoubleArrayToRows implements Function<Tuple2<Long, Iterable<Tuple2<Long, Double[]>>>, Row> {
		private static final long serialVersionUID = 4441184411670316972L;
		int bclen; long clen;
		public ConvertDoubleArrayToRows(long clen, int bclen) {
			this.clen = clen;
			this.bclen = bclen;
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
			// TODO: Make this check: It is quite possible the matrix characteristics are not inferred
//			if(clen != sizeOfPartialRows) {
//				throw new Exception("Incorrect number of columns in the row:" + clen + " != " + sizeOfPartialRows);
//			}
			
			// Insert first row as row index
			Double[] row = new Double[sizeOfPartialRows + 1];
			long rowIndex = arg0._1;
			row[0] = new Double(rowIndex);
			for(long columnBlockIndex = 1; columnBlockIndex <= partialRows.size(); columnBlockIndex++) {
				if(partialRows.containsKey(columnBlockIndex)) {
					Double [] array = partialRows.get(columnBlockIndex);
					for(int i = 0; i < array.length; i++) {
						row[(int) ((columnBlockIndex-1)*bclen + i) + 1] = array[i];
					}
				}
				else {
					throw new Exception("The block for column index " + columnBlockIndex + " is missing. Make sure the last instruction is not returning empty blocks");
				}
			}
			
			Object[] row_fields = row;
			return RowFactory.create(row_fields);
		}

		
		
	}
	
}
