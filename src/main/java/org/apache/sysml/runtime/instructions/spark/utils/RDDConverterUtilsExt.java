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

	
	public static DataFrame projectColumns(DataFrame df, ArrayList<String> columns) throws DMLRuntimeException {
		ArrayList<String> columnToSelect = new ArrayList<String>();
		for(int i = 1; i < columns.size(); i++) {
			columnToSelect.add(columns.get(i));
		}
		return df.select(columns.get(0), scala.collection.JavaConversions.asScalaBuffer(columnToSelect).toList());
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
}
