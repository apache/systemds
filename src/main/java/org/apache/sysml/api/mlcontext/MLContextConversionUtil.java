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

package org.apache.sysml.api.mlcontext;

import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.MLContextProxy;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertStringToLongTextPair;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockPairFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyTextInputFunction;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.DataFrameAnalysisFunction;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.DataFrameToBinaryBlockFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.collection.JavaConversions;
import scala.reflect.ClassTag;

/**
 * Utility class containing methods to perform data conversions.
 *
 */
public class MLContextConversionUtil {

	/**
	 * Convert a two-dimensional double array to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param doubleMatrix
	 *            matrix of double values
	 * @return the two-dimensional double matrix converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject doubleMatrixToMatrixObject(String variableName, double[][] doubleMatrix) {
		return doubleMatrixToMatrixObject(variableName, doubleMatrix, null);
	}

	/**
	 * Convert a two-dimensional double array to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param doubleMatrix
	 *            matrix of double values
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the two-dimensional double matrix converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject doubleMatrixToMatrixObject(String variableName, double[][] doubleMatrix,
			MatrixMetadata matrixMetadata) {
		try {
			MatrixBlock matrixBlock = DataConverter.convertToMatrixBlock(doubleMatrix);
			MatrixCharacteristics matrixCharacteristics;
			if (matrixMetadata != null) {
				matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
			} else {
				matrixCharacteristics = new MatrixCharacteristics(matrixBlock.getNumRows(), matrixBlock.getNumColumns(),
						MLContextUtil.defaultBlockSize(), MLContextUtil.defaultBlockSize());
			}

			MatrixFormatMetaData meta = new MatrixFormatMetaData(matrixCharacteristics,
					OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE,
					MLContextUtil.scratchSpace() + "/" + variableName, meta);
			matrixObject.acquireModify(matrixBlock);
			matrixObject.release();
			return matrixObject;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception converting double[][] array to MatrixObject", e);
		}
	}

	/**
	 * Convert a matrix at a URL to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param url
	 *            the URL to a matrix (in CSV or IJV format)
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the matrix at a URL converted to a {@code MatrixObject}
	 */
	public static MatrixObject urlToMatrixObject(String variableName, URL url, MatrixMetadata matrixMetadata) {
		try {
			InputStream is = url.openStream();
			List<String> lines = IOUtils.readLines(is);
			MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
			SparkContext sparkContext = activeMLContext.getSparkContext();
			@SuppressWarnings("resource")
			JavaSparkContext javaSparkContext = new JavaSparkContext(sparkContext);
			JavaRDD<String> javaRDD = javaSparkContext.parallelize(lines);
			if ((matrixMetadata == null) || (matrixMetadata.getMatrixFormat() == MatrixFormat.CSV)) {
				MatrixObject matrixObject = javaRDDStringCSVToMatrixObject(variableName, javaRDD, matrixMetadata);
				return matrixObject;
			} else if (matrixMetadata.getMatrixFormat() == MatrixFormat.IJV) {
				MatrixObject matrixObject = javaRDDStringIJVToMatrixObject(variableName, javaRDD, matrixMetadata);
				return matrixObject;
			}
			return null;
		} catch (Exception e) {
			throw new MLContextException("Exception converting URL to MatrixObject", e);
		}
	}

	/**
	 * Convert a {@code MatrixBlock} to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param matrixBlock
	 *            matrix as a MatrixBlock
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code MatrixBlock} converted to a {@code MatrixObject}
	 */
	public static MatrixObject matrixBlockToMatrixObject(String variableName, MatrixBlock matrixBlock,
			MatrixMetadata matrixMetadata) {
		try {
			MatrixCharacteristics matrixCharacteristics;
			if (matrixMetadata != null) {
				matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
			} else {
				matrixCharacteristics = new MatrixCharacteristics();
			}
			MatrixFormatMetaData mtd = new MatrixFormatMetaData(matrixCharacteristics, OutputInfo.BinaryBlockOutputInfo,
					InputInfo.BinaryBlockInputInfo);
			MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE,
					MLContextUtil.scratchSpace() + "/" + variableName, mtd);
			matrixObject.acquireModify(matrixBlock);
			matrixObject.release();
			return matrixObject;
		} catch (CacheException e) {
			throw new MLContextException("Exception converting MatrixBlock to MatrixObject", e);
		}
	}

	/**
	 * Convert a {@code FrameBlock} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param frameBlock
	 *            frame as a FrameBlock
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code FrameBlock} converted to a {@code FrameObject}
	 */
	public static FrameObject frameBlockToFrameObject(String variableName, FrameBlock frameBlock,
			MatrixMetadata matrixMetadata) {
		try {
			MatrixCharacteristics matrixCharacteristics;
			if (matrixMetadata != null) {
				matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
			} else {
				matrixCharacteristics = new MatrixCharacteristics();
			}
			MatrixFormatMetaData mtd = new MatrixFormatMetaData(matrixCharacteristics, OutputInfo.BinaryBlockOutputInfo,
					InputInfo.BinaryBlockInputInfo);
			FrameObject frameObject = new FrameObject(MLContextUtil.scratchSpace() + "/" + variableName, mtd);
			frameObject.acquireModify(frameBlock);
			frameObject.release();
			return frameObject;
		} catch (CacheException e) {
			throw new MLContextException("Exception converting FrameBlock to FrameObject", e);
		}
	}

	/**
	 * Convert a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} to a
	 * {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} representation
	 *            of a binary-block matrix
	 * @return the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 *         converted to a {@code MatrixObject}
	 */
	public static MatrixObject binaryBlocksToMatrixObject(String variableName,
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks) {
		return binaryBlocksToMatrixObject(variableName, binaryBlocks, null);
	}

	/**
	 * Convert a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} to a
	 * {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} representation
	 *            of a binary-block matrix
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 *         converted to a {@code MatrixObject}
	 */
	public static MatrixObject binaryBlocksToMatrixObject(String variableName,
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks, MatrixMetadata matrixMetadata) {

		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}

		JavaPairRDD<MatrixIndexes, MatrixBlock> javaPairRdd = binaryBlocks.mapToPair(new CopyBlockPairFunction());

		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE,
				MLContextUtil.scratchSpace() + "/" + "temp_" + System.nanoTime(), new MatrixFormatMetaData(
						matrixCharacteristics, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		matrixObject.setRDDHandle(new RDDObject(javaPairRdd, variableName));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a
	 * {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation
	 *            of a binary-block frame
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame
	 *         converted to a {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(String variableName,
			JavaPairRDD<Long, FrameBlock> binaryBlocks) {
		return binaryBlocksToFrameObject(variableName, binaryBlocks, null);
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a
	 * {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation
	 *            of a binary-block frame
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame
	 *         converted to a {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(String variableName,
			JavaPairRDD<Long, FrameBlock> binaryBlocks, MatrixMetadata matrixMetadata) {

		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}

		MatrixFormatMetaData mtd = new MatrixFormatMetaData(matrixCharacteristics,
				OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		FrameObject frameObject = new FrameObject(MLContextUtil.scratchSpace() + "/" + "temp_"			//TODO valuetype and variab le name?
				+ System.nanoTime() + variableName, mtd);
		frameObject.setRDDHandle(new RDDObject(binaryBlocks, variableName));
		return frameObject;
	}

	/**
	 * Convert a {@code DataFrame} to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject dataFrameToMatrixObject(String variableName, DataFrame dataFrame) {
		return dataFrameToMatrixObject(variableName, dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a {@code MatrixObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject dataFrameToMatrixObject(String variableName, DataFrame dataFrame,
			MatrixMetadata matrixMetadata) {
		if (matrixMetadata == null) {
			matrixMetadata = new MatrixMetadata();
		}
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlock = MLContextConversionUtil.dataFrameToBinaryBlocks(dataFrame,
				matrixMetadata);
		MatrixObject matrixObject = MLContextConversionUtil.binaryBlocksToMatrixObject(variableName, binaryBlock,
				matrixMetadata);
		return matrixObject;
	}

	/**
	 * Convert a {@code DataFrame} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code FrameObject}
	 * @throws DMLRuntimeException 
	 */
	public static FrameObject dataFrameToFramexObject(String variableName, DataFrame dataFrame) throws DMLRuntimeException {
		return dataFrameToFrameObject(variableName, dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code DataFrame} frame converted to a converted to a
	 *         {@code FrameObject}
	 * @throws DMLRuntimeException 
	 */
	public static FrameObject dataFrameToFrameObject(String variableName, DataFrame dataFrame,
			MatrixMetadata matrixMetadata) throws DMLRuntimeException {
		if (matrixMetadata == null) {
			matrixMetadata = new MatrixMetadata();
		}

		JavaSparkContext javaSparkContext = getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContext());
		boolean containsID = isDataFrameWithIDColumn(matrixMetadata); 
		MatrixCharacteristics matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		JavaPairRDD<Long, FrameBlock> binaryBlock = 
				FrameRDDConverterUtils.dataFrameToBinaryBlock(javaSparkContext, dataFrame, 
						matrixCharacteristics, containsID);

		FrameObject frameObject = MLContextConversionUtil.binaryBlocksToFrameObject(variableName, binaryBlock,
				matrixMetadata);
		return frameObject;
	}

	/**
	 * Convert a {@code DataFrame} to a
	 * {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} binary-block matrix.
	 * 
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @return the {@code DataFrame} matrix converted to a
	 *         {@code JavaPairRDD<MatrixIndexes,
	 *         MatrixBlock>} binary-block matrix
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlocks(DataFrame dataFrame) {
		return dataFrameToBinaryBlocks(dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a
	 * {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} binary-block matrix.
	 * 
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code DataFrame} matrix converted to a
	 *         {@code JavaPairRDD<MatrixIndexes,
	 *         MatrixBlock>} binary-block matrix
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToBinaryBlocks(DataFrame dataFrame,
			MatrixMetadata matrixMetadata) {

		determineMatrixFormatIfNeeded(dataFrame, matrixMetadata);

		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
			if (matrixCharacteristics == null) {
				matrixCharacteristics = new MatrixCharacteristics();
			}
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}

		if (isDataFrameWithIDColumn(matrixMetadata)) {
			dataFrame = dataFrame.sort("ID").drop("ID");
		}

		boolean isVectorBasedDataFrame = isVectorBasedDataFrame(matrixMetadata);

		determineDataFrameDimensionsIfNeeded(dataFrame, matrixCharacteristics, isVectorBasedDataFrame);
		if (matrixMetadata != null) {
			// so external reference can be updated with the metadata
			matrixMetadata.setMatrixCharacteristics(matrixCharacteristics);
		}

		JavaRDD<Row> javaRDD = dataFrame.javaRDD();
		JavaPairRDD<Row, Long> prepinput = javaRDD.zipWithIndex();
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = prepinput
				.mapPartitionsToPair(new DataFrameToBinaryBlockFunction(matrixCharacteristics, isVectorBasedDataFrame));
		out = RDDAggregateUtils.mergeByKey(out);
		return out;
	}

	/**
	 * If the MatrixFormat of the DataFrame has not been explicitly specified,
	 * attempt to determine the proper MatrixFormat.
	 * 
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixMetadata
	 *            the matrix metadata, if available
	 */
	public static void determineMatrixFormatIfNeeded(DataFrame dataFrame, MatrixMetadata matrixMetadata) {
		MatrixFormat matrixFormat = matrixMetadata.getMatrixFormat();
		if (matrixFormat != null) {
			return;
		}
		StructType schema = dataFrame.schema();
		boolean hasID = false;
		try {
			schema.fieldIndex("ID");
			hasID = true;
		} catch (IllegalArgumentException iae) {
		}
		Row firstRow = dataFrame.first();
		MatrixFormat mf = null;
		if (hasID) {
			Object object = firstRow.get(1);
			if (object instanceof Vector) {
				mf = MatrixFormat.DF_VECTOR_WITH_ID_COLUMN;
			} else {
				mf = MatrixFormat.DF_DOUBLES_WITH_ID_COLUMN;
			}
		} else {
			Object object = firstRow.get(0);
			if (object instanceof Vector) {
				mf = MatrixFormat.DF_VECTOR_WITH_NO_ID_COLUMN;
			} else {
				mf = MatrixFormat.DF_DOUBLES_WITH_NO_ID_COLUMN;
			}
		}
		if (mf == null) {
			throw new MLContextException("DataFrame format not recognized as an accepted SystemML MatrixFormat");
		}
		matrixMetadata.setMatrixFormat(mf);
	}

	/**
	 * Return whether or not the DataFrame has an ID column.
	 * 
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return {@code true} if the DataFrame has an ID column, {@code false}
	 *         otherwise.
	 */
	public static boolean isDataFrameWithIDColumn(MatrixMetadata matrixMetadata) {
		if (matrixMetadata == null) {
			return false;
		}
		MatrixFormat matrixFormat = matrixMetadata.getMatrixFormat();
		if (matrixFormat == null) {
			return false;
		}
		return matrixFormat.hasIDColumn();
	}

	/**
	 * Return whether or not the DataFrame is vector-based.
	 * 
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return {@code true} if the DataFrame is vector-based, {@code false}
	 *         otherwise.
	 */
	public static boolean isVectorBasedDataFrame(MatrixMetadata matrixMetadata) {
		if (matrixMetadata == null) {
			return false;
		}
		MatrixFormat matrixFormat = matrixMetadata.getMatrixFormat();
		if (matrixFormat == null) {
			return false;
		}
		return matrixFormat.isVectorBased();
	}

	/**
	 * If the {@code DataFrame} dimensions aren't present in the
	 * {@code MatrixCharacteristics} metadata, determine the dimensions and
	 * place them in the {@code MatrixCharacteristics} metadata.
	 * 
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixCharacteristics
	 *            the matrix metadata
	 * @param vectorBased
	 *            is the DataFrame vector-based
	 */
	public static void determineDataFrameDimensionsIfNeeded(DataFrame dataFrame,
			MatrixCharacteristics matrixCharacteristics, boolean vectorBased) {
		if (!matrixCharacteristics.dimsKnown(true)) {
			MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
			SparkContext sparkContext = activeMLContext.getSparkContext();
			@SuppressWarnings("resource")
			JavaSparkContext javaSparkContext = new JavaSparkContext(sparkContext);

			Accumulator<Double> aNnz = javaSparkContext.accumulator(0L);
			JavaRDD<Row> javaRDD = dataFrame.javaRDD().map(new DataFrameAnalysisFunction(aNnz, vectorBased));
			long numRows = javaRDD.count();
			long numColumns;
			if (vectorBased) {
				Vector v = (Vector) javaRDD.first().get(0);
				numColumns = v.size();
			} else {
				numColumns = dataFrame.columns().length;
			}

			long numNonZeros = UtilFunctions.toLong(aNnz.value());
			matrixCharacteristics.set(numRows, numColumns, matrixCharacteristics.getRowsPerBlock(),
					matrixCharacteristics.getColsPerBlock(), numNonZeros);
		}
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code MatrixObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringCSVToMatrixObject(String variableName, JavaRDD<String> javaRDD) {
		return javaRDDStringCSVToMatrixObject(variableName, javaRDD, null);
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code MatrixObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringCSVToMatrixObject(String variableName, JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}
		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, null,
				new MatrixFormatMetaData(matrixCharacteristics, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
		JavaPairRDD<LongWritable, Text> javaPairRDD2 = javaPairRDD.mapToPair(new CopyTextInputFunction());
		matrixObject.setRDDHandle(new RDDObject(javaPairRDD2, variableName));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code FrameObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringCSVToFrameObject(String variableName, JavaRDD<String> javaRDD) {
		return javaRDDStringCSVToFrameObject(variableName, javaRDD, null);
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code FrameObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringCSVToFrameObject(String variableName, JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}
		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());
		
		JavaSparkContext jsc = getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContext());

		FrameObject frameObject = new FrameObject(null, new MatrixFormatMetaData(matrixCharacteristics, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			rdd = FrameRDDConverterUtils.csvToBinaryBlock(jsc, javaPairRDDText, matrixCharacteristics, false, ",", false, -1);
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			return null;
		}
		frameObject.setRDDHandle(new RDDObject(rdd, variableName));
		return frameObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in IJV format to a {@code MatrixObject}
	 * . Note that metadata is required for IJV format.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringIJVToMatrixObject(String variableName, JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}
		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(
				matrixCharacteristics, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		JavaPairRDD<LongWritable, Text> javaPairRDD2 = javaPairRDD.mapToPair(new CopyTextInputFunction());
		matrixObject.setRDDHandle(new RDDObject(javaPairRDD2, variableName));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in IJV format to a {@code FrameObject}
	 * . Note that metadata is required for IJV format.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringIJVToFrameObject(String variableName, JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics matrixCharacteristics;
		if (matrixMetadata != null) {
			matrixCharacteristics = matrixMetadata.asMatrixCharacteristics();
		} else {
			matrixCharacteristics = new MatrixCharacteristics();
		}
		
		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());
		
		JavaSparkContext jsc = getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContext());

		FrameObject frameObject = new FrameObject(null, new MatrixFormatMetaData(matrixCharacteristics, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			List<ValueType> lschema = null;
			if(lschema == null)
				lschema = Collections.nCopies((int)matrixCharacteristics.getCols(), ValueType.STRING);
			rdd = FrameRDDConverterUtils.textCellToBinaryBlock(jsc, javaPairRDDText, matrixCharacteristics, lschema);
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			return null;
		}
		frameObject.setRDDHandle(new RDDObject(rdd, variableName));
		return frameObject;
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code MatrixObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param rdd
	 *            the RDD of strings
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringCSVToMatrixObject(String variableName, RDD<String> rdd) {
		return rddStringCSVToMatrixObject(variableName, rdd, null);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code MatrixObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringCSVToMatrixObject(String variableName, RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringCSVToMatrixObject(variableName, javaRDD, matrixMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code FrameObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param rdd
	 *            the RDD of strings
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringCSVToFrameObject(String variableName, RDD<String> rdd) {
		return rddStringCSVToFrameObject(variableName, rdd, null);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code FrameObject}
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringCSVToFrameObject(String variableName, RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringCSVToFrameObject(variableName, javaRDD, matrixMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in IJV format to a {@code MatrixObject}.
	 * Note that metadata is required for IJV format.
	 * 
	 * @param variableName
	 *            name of the variable associated with the matrix
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringIJVToMatrixObject(String variableName, RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringIJVToMatrixObject(variableName, javaRDD, matrixMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in IJV format to a {@code FrameObject}.
	 * Note that metadata is required for IJV format.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringIJVToFrameObject(String variableName, RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringIJVToFrameObject(variableName, javaRDD, matrixMetadata);
	}

	/**
	 * Convert an {@code BinaryBlockMatrix} to a {@code JavaRDD<String>} in IVJ
	 * format.
	 * 
	 * @param binaryBlockMatrix
	 *            the {@code BinaryBlockMatrix}
	 * @return the {@code BinaryBlockMatrix} converted to a
	 *         {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> binaryBlockMatrixToJavaRDDStringIJV(BinaryBlockMatrix binaryBlockMatrix) {
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlock = binaryBlockMatrix.getBinaryBlocks();
		MatrixCharacteristics matrixCharacteristics = binaryBlockMatrix.getMatrixCharacteristics();
		try {
			JavaRDD<String> javaRDDString = RDDConverterUtilsExt.binaryBlockToStringRDD(binaryBlock,
					matrixCharacteristics, "text");
			return javaRDDString;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception converting BinaryBlockMatrix to JavaRDD<String> (ijv)", e);
		}
	}

	/**
	 * Convert an {@code BinaryBlockMatrix} to a {@code RDD<String>} in IVJ
	 * format.
	 * 
	 * @param binaryBlockMatrix
	 *            the {@code BinaryBlockMatrix}
	 * @return the {@code BinaryBlockMatrix} converted to a {@code RDD<String>}
	 */
	public static RDD<String> binaryBlockMatrixToRDDStringIJV(BinaryBlockMatrix binaryBlockMatrix) {
		JavaRDD<String> javaRDD = binaryBlockMatrixToJavaRDDStringIJV(binaryBlockMatrix);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);
		return rdd;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code JavaRDD<String>} in CSV
	 * format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> matrixObjectToJavaRDDStringCSV(MatrixObject matrixObject) {
		List<String> list = matrixObjectToListStringCSV(matrixObject);

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
		SparkContext sc = activeMLContext.getSparkContext();
		@SuppressWarnings("resource")
		JavaSparkContext jsc = new JavaSparkContext(sc);
		JavaRDD<String> javaRDDStringCSV = jsc.parallelize(list);
		return javaRDDStringCSV;
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaRDD<String>} in CSV
	 * format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> frameObjectToJavaRDDStringCSV(FrameObject frameObject, String delimiter) {
		List<String> list = frameObjectToListStringCSV(frameObject, delimiter);

		JavaSparkContext jsc = getJavaSparkContext((MLContext)MLContextProxy.getActiveMLContext());
		JavaRDD<String> javaRDDStringCSV = jsc.parallelize(list);
		return javaRDDStringCSV;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code JavaRDD<String>} in IJV
	 * format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> matrixObjectToJavaRDDStringIJV(MatrixObject matrixObject) {
		List<String> list = matrixObjectToListStringIJV(matrixObject);

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
		SparkContext sc = activeMLContext.getSparkContext();
		@SuppressWarnings("resource")
		JavaSparkContext jsc = new JavaSparkContext(sc);
		JavaRDD<String> javaRDDStringCSV = jsc.parallelize(list);
		return javaRDDStringCSV;
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaRDD<String>} in IJV
	 * format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> frameObjectToJavaRDDStringIJV(FrameObject frameObject) {
		List<String> list = frameObjectToListStringIJV(frameObject);

		JavaSparkContext jsc = getJavaSparkContext((MLContext)MLContextProxy.getActiveMLContext());
		JavaRDD<String> javaRDDStringIJV = jsc.parallelize(list);
		return javaRDDStringIJV;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code RDD<String>} in IJV format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code RDD<String>}
	 */
	public static RDD<String> matrixObjectToRDDStringIJV(MatrixObject matrixObject) {

		// NOTE: The following works when called from Java but does not
		// currently work when called from Spark Shell (when you call
		// collect() on the RDD<String>).
		//
		// JavaRDD<String> javaRDD = jsc.parallelize(list);
		// RDD<String> rdd = JavaRDD.toRDD(javaRDD);
		//
		// Therefore, we call parallelize() on the SparkContext rather than
		// the JavaSparkContext to produce the RDD<String> for Scala.

		List<String> list = matrixObjectToListStringIJV(matrixObject);

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
		SparkContext sc = activeMLContext.getSparkContext();
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		RDD<String> rddString = sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
		return rddString;
	}

	/**
	 * Convert a {@code FrameObject} to a {@code RDD<String>} in IJV format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code RDD<String>}
	 */
	public static RDD<String> frameObjectToRDDStringIJV(FrameObject frameObject) {

		// NOTE: The following works when called from Java but does not
		// currently work when called from Spark Shell (when you call
		// collect() on the RDD<String>).
		//
		// JavaRDD<String> javaRDD = jsc.parallelize(list);
		// RDD<String> rdd = JavaRDD.toRDD(javaRDD);
		//
		// Therefore, we call parallelize() on the SparkContext rather than
		// the JavaSparkContext to produce the RDD<String> for Scala.

		List<String> list = frameObjectToListStringIJV(frameObject);

		SparkContext sc = getSparkContext((MLContext) MLContextProxy.getActiveMLContext());
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		RDD<String> rddString = sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
		return rddString;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code RDD<String>} in CSV format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code RDD<String>}
	 */
	public static RDD<String> matrixObjectToRDDStringCSV(MatrixObject matrixObject) {

		// NOTE: The following works when called from Java but does not
		// currently work when called from Spark Shell (when you call
		// collect() on the RDD<String>).
		//
		// JavaRDD<String> javaRDD = jsc.parallelize(list);
		// RDD<String> rdd = JavaRDD.toRDD(javaRDD);
		//
		// Therefore, we call parallelize() on the SparkContext rather than
		// the JavaSparkContext to produce the RDD<String> for Scala.

		List<String> list = matrixObjectToListStringCSV(matrixObject);

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
		SparkContext sc = activeMLContext.getSparkContext();
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		RDD<String> rddString = sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
		return rddString;
	}

	/**
	 * Convert a {@code FrameObject} to a {@code RDD<String>} in CSV format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code RDD<String>}
	 */
	public static RDD<String> frameObjectToRDDStringCSV(FrameObject frameObject, String delimiter) {

		// NOTE: The following works when called from Java but does not
		// currently work when called from Spark Shell (when you call
		// collect() on the RDD<String>).
		//
		// JavaRDD<String> javaRDD = jsc.parallelize(list);
		// RDD<String> rdd = JavaRDD.toRDD(javaRDD);
		//
		// Therefore, we call parallelize() on the SparkContext rather than
		// the JavaSparkContext to produce the RDD<String> for Scala.

		List<String> list = frameObjectToListStringCSV(frameObject, delimiter);

		SparkContext sc = getSparkContext((MLContext) MLContextProxy.getActiveMLContext());
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		RDD<String> rddString = sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
		return rddString;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code List<String>} in CSV format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code List<String>}
	 */
	public static List<String> matrixObjectToListStringCSV(MatrixObject matrixObject) {
		try {
			MatrixBlock mb = matrixObject.acquireRead();

			int rows = mb.getNumRows();
			int cols = mb.getNumColumns();
			List<String> list = new ArrayList<String>();

			if (mb.getNonZeros() > 0) {
				if (mb.isInSparseFormat()) {
					Iterator<IJV> iter = mb.getSparseBlockIterator();
					int prevCellRow = -1;
					StringBuilder sb = null;
					while (iter.hasNext()) {
						IJV cell = iter.next();
						int i = cell.getI();
						double v = cell.getV();
						if (i > prevCellRow) {
							if (sb == null) {
								sb = new StringBuilder();
							} else {
								list.add(sb.toString());
								sb = new StringBuilder();
							}
							sb.append(v);
							prevCellRow = i;
						} else if (i == prevCellRow) {
							sb.append(",");
							sb.append(v);
						}
					}
					if (sb != null) {
						list.add(sb.toString());
					}
				} else {
					for (int i = 0; i < rows; i++) {
						StringBuilder sb = new StringBuilder();
						for (int j = 0; j < cols; j++) {
							if (j > 0) {
								sb.append(",");
							}
							sb.append(mb.getValueDenseUnsafe(i, j));
						}
						list.add(sb.toString());
					}
				}
			}

			matrixObject.release();
			return list;
		} catch (CacheException e) {
			throw new MLContextException("Cache exception while converting matrix object to List<String> CSV format",
					e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a {@code List<String>} in CSV format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code List<String>}
	 */
	public static List<String> frameObjectToListStringCSV(FrameObject frameObject, String delimiter) {
		try {
			FrameBlock fb = frameObject.acquireRead();

			int rows = fb.getNumRows();
			int cols = fb.getNumColumns();
			List<String> list = new ArrayList<String>();

			for (int i = 0; i < rows; i++) {
				StringBuilder sb = new StringBuilder();
				for (int j = 0; j < cols; j++) {
					if (j > 0) {
						sb.append(delimiter);
					}
					sb.append(fb.get(i, j));
				}
				list.add(sb.toString());
			}

			frameObject.release();
			return list;
		} catch (CacheException e) {
			throw new MLContextException("Cache exception while converting frame object to List<String> CSV format", e);
		}
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code List<String>} in IJV format.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code List<String>}
	 */
	public static List<String> matrixObjectToListStringIJV(MatrixObject matrixObject) {
		try {
			MatrixBlock mb = matrixObject.acquireRead();

			int rows = mb.getNumRows();
			int cols = mb.getNumColumns();
			List<String> list = new ArrayList<String>();

			if (mb.getNonZeros() > 0) {
				if (mb.isInSparseFormat()) {
					Iterator<IJV> iter = mb.getSparseBlockIterator();
					StringBuilder sb = null;
					while (iter.hasNext()) {
						IJV cell = iter.next();
						sb = new StringBuilder();
						sb.append(cell.getI() + 1);
						sb.append(" ");
						sb.append(cell.getJ() + 1);
						sb.append(" ");
						sb.append(cell.getV());
						list.add(sb.toString());
					}
				} else {
					StringBuilder sb = null;
					for (int i = 0; i < rows; i++) {
						sb = new StringBuilder();
						for (int j = 0; j < cols; j++) {
							sb = new StringBuilder();
							sb.append(i + 1);
							sb.append(" ");
							sb.append(j + 1);
							sb.append(" ");
							sb.append(mb.getValueDenseUnsafe(i, j));
							list.add(sb.toString());
						}
					}
				}
			}

			matrixObject.release();
			return list;
		} catch (CacheException e) {
			throw new MLContextException("Cache exception while converting matrix object to List<String> IJV format",
					e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a {@code List<String>} in IJV format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code List<String>}
	 */
	public static List<String> frameObjectToListStringIJV(FrameObject frameObject) {
		try {
			FrameBlock fb = frameObject.acquireRead();

			int rows = fb.getNumRows();
			int cols = fb.getNumColumns();
			List<String> list = new ArrayList<String>();

			StringBuilder sb = null;
			for (int i = 0; i < rows; i++) {
				sb = new StringBuilder();
				for (int j = 0; j < cols; j++) {
					sb = new StringBuilder();
					sb.append(i + 1);
					sb.append(" ");
					sb.append(j + 1);
					sb.append(" ");
					sb.append(fb.get(i, j));
					list.add(sb.toString());
				}
			}

			frameObject.release();
			return list;
		} catch (CacheException e) {
			throw new MLContextException("Cache exception while converting frame object to List<String> IJV format", e);
		}
	}

	/**
	 * Convert a {@code MatrixObject} to a two-dimensional double array.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code double[][]}
	 */
	public static double[][] matrixObjectToDoubleMatrix(MatrixObject matrixObject) {
		try {
			MatrixBlock mb = matrixObject.acquireRead();
			double[][] matrix = DataConverter.convertToDoubleMatrix(mb);
			matrixObject.release();
			return matrix;
		} catch (CacheException e) {
			throw new MLContextException("Cache exception while converting matrix object to double matrix", e);
		}
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code DataFrame}.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code MatrixObject} converted to a {@code DataFrame}
	 */
	public static DataFrame matrixObjectToDataFrame(MatrixObject matrixObject,
			SparkExecutionContext sparkExecutionContext, boolean isVectorDF) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlockMatrix = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sparkExecutionContext
					.getRDDHandleForMatrixObject(matrixObject, InputInfo.BinaryBlockInputInfo);
			MatrixCharacteristics matrixCharacteristics = matrixObject.getMatrixCharacteristics();

			MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContext();
			SparkContext sc = activeMLContext.getSparkContext();
			SQLContext sqlContext = new SQLContext(sc);
			DataFrame df = null;
			if (isVectorDF) {
				df = RDDConverterUtilsExt.binaryBlockToVectorDataFrame(binaryBlockMatrix, matrixCharacteristics,
						sqlContext);
			} else {
				df = RDDConverterUtilsExt.binaryBlockToDataFrame(binaryBlockMatrix, matrixCharacteristics, sqlContext);
			}

			return df;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting matrix object to DataFrame", e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a {@code DataFrame}.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code FrameObject} converted to a {@code DataFrame}
	 */
	public static DataFrame frameObjectToDataFrame(FrameObject frameObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<Long, FrameBlock> binaryBlockFrame = (JavaPairRDD<Long, FrameBlock>) sparkExecutionContext
					.getRDDHandleForFrameObject(frameObject, InputInfo.BinaryBlockInputInfo);
			MatrixCharacteristics matrixCharacteristics = frameObject.getMatrixCharacteristics();

			JavaSparkContext jsc = getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContext());

			return FrameRDDConverterUtils.binaryBlockToDataFrame(binaryBlockFrame, matrixCharacteristics, jsc);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting frame object to DataFrame", e);
		}
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code BinaryBlockMatrix}.
	 * 
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code MatrixObject} converted to a {@code BinaryBlockMatrix}
	 */
	public static BinaryBlockMatrix matrixObjectToBinaryBlockMatrix(MatrixObject matrixObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlock = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sparkExecutionContext
					.getRDDHandleForMatrixObject(matrixObject, InputInfo.BinaryBlockInputInfo);
			MatrixCharacteristics matrixCharacteristics = matrixObject.getMatrixCharacteristics();
			BinaryBlockMatrix binaryBlockMatrix = new BinaryBlockMatrix(binaryBlock, matrixCharacteristics);
			return binaryBlockMatrix;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting matrix object to BinaryBlockMatrix", e);
		}
	}
	
	public static SparkContext getSparkContext(MLContext mlContext)
	{
		return mlContext.getSparkContext();
	}

	public static JavaSparkContext getJavaSparkContext(MLContext mlContext)
	{
		return new JavaSparkContext(mlContext.getSparkContext());
	}

}
