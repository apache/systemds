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

import org.apache.hadoop.io.Text;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.util.NumericParser;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixCell;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.mapred.ReblockBuffer;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.util.FastStringTokenizer;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * NOTE: These are experimental converter utils. Once thoroughly tested, they
 * can be moved to RDDConverterUtils.
 */
@SuppressWarnings("unused")
public class RDDConverterUtilsExt
{
	public enum RDDConverterTypes {
		TEXT_TO_MATRIX_CELL,
		MATRIXENTRY_TO_MATRIXCELL
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> coordinateMatrixToBinaryBlock(JavaSparkContext sc,
		CoordinateMatrix input, DataCharacteristics mcIn, boolean outputEmptyBlocks)
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
		out = RDDAggregateUtils.mergeByKey(out, false);

		return out;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> coordinateMatrixToBinaryBlock(SparkContext sc,
			CoordinateMatrix input, DataCharacteristics mcIn, boolean outputEmptyBlocks) {
		return coordinateMatrixToBinaryBlock(new JavaSparkContext(sc), input, mcIn, true);
	}

	public static Dataset<Row> projectColumns(Dataset<Row> df, ArrayList<String> columns) {
		ArrayList<String> columnToSelect = new ArrayList<String>();
		for(int i = 1; i < columns.size(); i++) {
			columnToSelect.add(columns.get(i));
		}
		return df.select(columns.get(0), scala.collection.JavaConversions.asScalaBuffer(columnToSelect).toList());
	}

	public static MatrixBlock convertPy4JArrayToMB(byte [] data, long rlen, long clen) {
		return convertPy4JArrayToMB(data, (int)rlen, (int)clen, false);
	}

	public static MatrixBlock convertPy4JArrayToMB(byte [] data, int rlen, int clen) {
		return convertPy4JArrayToMB(data, rlen, clen, false);
	}

	public static MatrixBlock convertSciPyCOOToMB(byte [] data, byte [] row, byte [] col, long rlen, long clen, long nnz) {
		return convertSciPyCOOToMB(data, row, col, (int)rlen, (int)clen, (int)nnz);
	}

	public static MatrixBlock convertSciPyCOOToMB(byte [] data, byte [] row, byte [] col, int rlen, int clen, int nnz) {
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
			mb.setValue(rowIndex, colIndex, val);
		}
		mb.recomputeNonZeros();
		mb.examSparsity();
		return mb;
	}

	public static MatrixBlock convertPy4JArrayToMB(byte [] data, long rlen, long clen, boolean isSparse) {
		return convertPy4JArrayToMB(data, (int) rlen, (int) clen, isSparse);
	}

	public static MatrixBlock allocateDenseOrSparse(int rlen, int clen, boolean isSparse) {
		MatrixBlock ret = new MatrixBlock(rlen, clen, isSparse);
		ret.allocateBlock();
		return ret;
	}
	public static MatrixBlock allocateDenseOrSparse(long rlen, long clen, boolean isSparse) {
		if(rlen > Integer.MAX_VALUE || clen > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Dimensions of matrix are too large to be passed via NumPy/SciPy:" + rlen + " X " + clen);
		}
		return allocateDenseOrSparse(rlen, clen, isSparse);
	}

	public static void copyRowBlocks(MatrixBlock mb, int rowIndex, MatrixBlock ret, int numRowsPerBlock, int rlen, int clen) {
		copyRowBlocks(mb, (long)rowIndex, ret, (long)numRowsPerBlock, (long)rlen, (long)clen);
	}
	public static void copyRowBlocks(MatrixBlock mb, long rowIndex, MatrixBlock ret, int numRowsPerBlock, int rlen, int clen) {
		copyRowBlocks(mb, (long)rowIndex, ret, (long)numRowsPerBlock, (long)rlen, (long)clen);
	}
	public static void copyRowBlocks(MatrixBlock mb, int rowIndex, MatrixBlock ret, long numRowsPerBlock, long rlen, long clen) {
		copyRowBlocks(mb, (long)rowIndex, ret, (long)numRowsPerBlock, (long)rlen, (long)clen);
	}
	public static void copyRowBlocks(MatrixBlock mb, long rowIndex, MatrixBlock ret, long numRowsPerBlock, long rlen, long clen) {
		// TODO: Double-check if synchronization is required here.
		// synchronized (RDDConverterUtilsExt.class) {
			ret.copy((int)(rowIndex*numRowsPerBlock), (int)Math.min((rowIndex+1)*numRowsPerBlock-1, rlen-1), 0, (int)(clen-1), mb, false);
		// }
	}

	public static void postProcessAfterCopying(MatrixBlock ret) {
		ret.recomputeNonZeros();
		ret.examSparsity();
	}

	public static MatrixBlock convertPy4JArrayToMB(byte [] data, int rlen, int clen, boolean isSparse) {
		MatrixBlock mb = new MatrixBlock(rlen, clen, isSparse, -1);
		if(isSparse) {
			throw new DMLRuntimeException("Convertion to sparse format not supported");
		}
		else {
			long limit = rlen*clen;
			if( limit > Integer.MAX_VALUE )
				throw new DMLRuntimeException("Dense NumPy array of size " + limit + " cannot be converted to MatrixBlock");
			double [] denseBlock = new double[(int) limit];
			ByteBuffer buf = ByteBuffer.wrap(data);
			buf.order(ByteOrder.nativeOrder());
			for(int i = 0; i < rlen*clen; i++) {
				denseBlock[i] = buf.getDouble();
			}
			mb.init( denseBlock, rlen, clen );
		}
		mb.recomputeNonZeros();
		mb.examSparsity();
		return mb;
	}

	public static byte [] convertMBtoPy4JDenseArr(MatrixBlock mb) {
		byte [] ret = null;
		if(mb.isInSparseFormat()) {
			mb.sparseToDense();
		}

		long limit = mb.getNumRows()*mb.getNumColumns();
		int times = Double.SIZE / Byte.SIZE;
		if( limit > Integer.MAX_VALUE / times )
			throw new DMLRuntimeException("MatrixBlock of size " + limit + " cannot be converted to dense numpy array");
		ret = new byte[(int) (limit * times)];

		double [] denseBlock = mb.getDenseBlockValues();
		if(mb.isEmptyBlock()) {
			for(int i=0;i < limit;i++){
		        ByteBuffer.wrap(ret, i*times, times).order(ByteOrder.nativeOrder()).putDouble(0);
			}
		}
		else if(denseBlock == null) {
			throw new DMLRuntimeException("Error while dealing with empty blocks.");
		}
		else {
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
	 *
	 * @param df input data frame
	 * @param sparkSession the Spark Session
	 * @param nameOfCol name of index column
	 * @return new data frame
	 */
	public static Dataset<Row> addIDToDataFrame(Dataset<Row> df, SparkSession sparkSession, String nameOfCol) {
		StructField[] oldSchema = df.schema().fields();
		StructField[] newSchema = new StructField[oldSchema.length + 1];
		for(int i = 0; i < oldSchema.length; i++) {
			newSchema[i] = oldSchema[i];
		}
		newSchema[oldSchema.length] = DataTypes.createStructField(nameOfCol, DataTypes.DoubleType, false);
		// JavaRDD<Row> newRows = df.rdd().toJavaRDD().map(new AddRowID());
		JavaRDD<Row> newRows = df.rdd().toJavaRDD().zipWithIndex().map(new AddRowID());
		return sparkSession.createDataFrame(newRows, new StructType(newSchema));
	}

	private static class MatrixEntryToBinaryBlockFunction implements PairFlatMapFunction<Iterator<MatrixEntry>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 4907483236186747224L;

		private IJVToBinaryBlockFunctionHelper helper = null;
		public MatrixEntryToBinaryBlockFunction(DataCharacteristics mc) {
			helper = new IJVToBinaryBlockFunctionHelper(mc);
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<MatrixEntry> arg0) throws Exception {
			return helper.convertToBinaryBlock(arg0, RDDConverterTypes.MATRIXENTRY_TO_MATRIXCELL).iterator();
		}

	}

	private static class IJVToBinaryBlockFunctionHelper implements Serializable {
		private static final long serialVersionUID = -7952801318564745821L;
		//internal buffer size (aligned w/ default matrix block size)
		private static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
		private int _bufflen = -1;

		private long _rlen = -1;
		private long _clen = -1;
		private int _blen = -1;

		public IJVToBinaryBlockFunctionHelper(DataCharacteristics mc) {
			if(!mc.dimsKnown())
				throw new DMLRuntimeException("The dimensions need to be known in given DataCharacteristics for given input RDD");
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_blen = mc.getBlocksize();
			_blen = mc.getBlocksize();
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
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _blen);

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

		private static void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret )
			throws IOException, DMLRuntimeException
		{
			rbuff.flushBufferToBinaryBlocks().stream() // prevent library dependencies
				.map(b -> SparkUtils.fromIndexedMatrixBlock(b)).forEach(b -> ret.add(b));
		}
	}

	/**
	 * Convert a dataframe of comma-separated string rows to a dataframe of
	 * ml.linalg.Vector rows.
	 *
	 * <p>
	 * Example input rows:<br>
	 *
	 * <code>
	 * ((1.2, 4.3, 3.4))<br>
	 * (1.2, 3.4, 2.2)<br>
	 * [[1.2, 34.3, 1.2, 1.25]]<br>
	 * [1.2, 3.4]<br>
	 * </code>
	 *
	 * @param sparkSession
	 *            Spark Session
	 * @param inputDF
	 *            dataframe of comma-separated row strings to convert to
	 *            dataframe of ml.linalg.Vector rows
	 * @return dataframe of ml.linalg.Vector rows
	 */
	public static Dataset<Row> stringDataFrameToVectorDataFrame(SparkSession sparkSession, Dataset<Row> inputDF)
	{
		StructField[] oldSchema = inputDF.schema().fields();
		StructField[] newSchema = new StructField[oldSchema.length];
		for (int i = 0; i < oldSchema.length; i++) {
			String colName = oldSchema[i].name();
			newSchema[i] = DataTypes.createStructField(colName, new VectorUDT(), true);
		}

		// converter
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
				// ((1.2, 4.3, 3.4)) or (1.2, 3.4, 2.2)
				// [[1.2, 34.3, 1.2, 1.2]] or [1.2, 3.4]
				Object[] fields = new Object[oldNumCols];
				ArrayList<Object> fieldsArr = new ArrayList<Object>();
				for (int i = 0; i < oldRow.length(); i++) {
					Object ci = oldRow.get(i);
					if (ci == null) {
						fieldsArr.add(null);
					} else if (ci instanceof String) {
						String cis = (String) ci;
						StringBuffer sb = new StringBuffer(cis.trim());
						for (int nid = 0; i < 2; i++) { // remove two level
														// nesting
							if ((sb.charAt(0) == '(' && sb.charAt(sb.length() - 1) == ')')
									|| (sb.charAt(0) == '[' && sb.charAt(sb.length() - 1) == ']')) {
								sb.deleteCharAt(0);
								sb.setLength(sb.length() - 1);
							}
						}
						// have the replace code
						String ncis = "[" + sb.toString().replaceAll(" *, *", ",") + "]";

						try {
							// ncis [ ] will always result in double array return type
							double[] doubles = (double[]) NumericParser.parse(ncis);
							Vector dense = Vectors.dense(doubles);
							fieldsArr.add(dense);
						} catch (Exception e) { // can't catch SparkException here in Java apparently
							throw new DMLRuntimeException("Error converting to double array. " + e.getMessage(), e);
						}

					} else {
						throw new DMLRuntimeException("Only String is supported");
					}
				}
				Row row = RowFactory.create(fieldsArr.toArray());
				return row;
			}
		}

		// output DF
		JavaRDD<Row> newRows = inputDF.rdd().toJavaRDD().zipWithIndex().map(new StringToVector());
		Dataset<Row> outDF = sparkSession.createDataFrame(newRows.rdd(), DataTypes.createStructType(newSchema));
		return outDF;
	}
}
