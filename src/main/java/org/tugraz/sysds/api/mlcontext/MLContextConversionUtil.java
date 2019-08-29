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

package org.tugraz.sysds.api.mlcontext;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.data.DatasetObject;
import org.tugraz.sysds.runtime.instructions.spark.data.RDDObject;
import org.tugraz.sysds.runtime.instructions.spark.functions.ConvertStringToLongTextPair;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyTextInputFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.IJV;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.data.Pair;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.collection.JavaConversions;
import scala.reflect.ClassTag;

import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

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
			MatrixCharacteristics mc = (matrixMetadata != null) ? matrixMetadata.asMatrixCharacteristics()
					: new MatrixCharacteristics(matrixBlock.getNumRows(), matrixBlock.getNumColumns(),
							ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());

			MatrixObject matrixObject = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
					new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));

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
	 * @param url
	 *            the URL to a matrix (in CSV or IJV format)
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the matrix at a URL converted to a {@code MatrixObject}
	 */
	public static MatrixObject urlToMatrixObject(URL url, MatrixMetadata matrixMetadata) {
		try {
			InputStream is = url.openStream();
			List<String> lines = IOUtils.readLines(is);
			JavaRDD<String> javaRDD = jsc().parallelize(lines);
			if ((matrixMetadata == null) || (matrixMetadata.getMatrixFormat() == MatrixFormat.CSV)) {
				return javaRDDStringCSVToMatrixObject(javaRDD, matrixMetadata);
			} else if (matrixMetadata.getMatrixFormat() == MatrixFormat.IJV) {
				return javaRDDStringIJVToMatrixObject(javaRDD, matrixMetadata);
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
			MatrixCharacteristics mc = (matrixMetadata != null) ? matrixMetadata.asMatrixCharacteristics()
					: new MatrixCharacteristics();
			MatrixObject matrixObject = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
					new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
			matrixObject.acquireModify(matrixBlock);
			matrixObject.release();
			return matrixObject;
		} catch (DMLRuntimeException e) {
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
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code FrameBlock} converted to a {@code FrameObject}
	 */
	public static FrameObject frameBlockToFrameObject(String variableName, FrameBlock frameBlock,
			FrameMetadata frameMetadata) {
		try {
			MatrixCharacteristics mc = (frameMetadata != null) ? frameMetadata.asMatrixCharacteristics()
					: new MatrixCharacteristics();
			MetaDataFormat mtd = new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo,
					InputInfo.BinaryBlockInputInfo);
			FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(), mtd,
					frameMetadata.getFrameSchema().getSchema().toArray(new ValueType[0]));
			frameObject.acquireModify(frameBlock);
			frameObject.release();
			return frameObject;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception converting FrameBlock to FrameObject", e);
		}
	}

	/**
	 * Convert a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} to a
	 * {@code MatrixObject}.
	 *
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} representation
	 *            of a binary-block matrix
	 * @return the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 *         converted to a {@code MatrixObject}
	 */
	public static MatrixObject binaryBlocksToMatrixObject(
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks) {
		return binaryBlocksToMatrixObject(binaryBlocks, null);
	}

	/**
	 * Convert a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} to a
	 * {@code MatrixObject}.
	 *
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} representation
	 *            of a binary-block matrix
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 *         converted to a {@code MatrixObject}
	 */
	public static MatrixObject binaryBlocksToMatrixObject(
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks, MatrixMetadata matrixMetadata) {
		return binaryBlocksToMatrixObject(binaryBlocks, matrixMetadata, true);
	}

	private static MatrixObject binaryBlocksToMatrixObject(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks,
		MatrixMetadata matrixMetadata, boolean copy) {

		MatrixCharacteristics mc = (matrixMetadata != null) ? matrixMetadata.asMatrixCharacteristics()
				: new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> javaPairRdd = SparkUtils.copyBinaryBlockMatrix(binaryBlocks, copy);
		MatrixObject matrixObject = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		matrixObject.setRDDHandle(new RDDObject(javaPairRdd));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} to a
	 * {@code MatrixBlock}
	 *
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} representation
	 *            of a binary-block matrix
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 *         converted to a {@code MatrixBlock}
	 */
	public static MatrixBlock binaryBlocksToMatrixBlock(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks,
			MatrixMetadata matrixMetadata) {
		try {
			MatrixBlock matrixBlock = SparkExecutionContext.toMatrixBlock(binaryBlocks,
					matrixMetadata.getNumRows().intValue(), matrixMetadata.getNumColumns().intValue(),
					matrixMetadata.getNumRowsPerBlock(), matrixMetadata.getNumColumnsPerBlock(),
					matrixMetadata.getNumNonZeros());
			return matrixBlock;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception converting binary blocks to MatrixBlock", e);
		}
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a {@code FrameObject}.
	 *
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation of a
	 *            binary-block frame
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(JavaPairRDD<Long, FrameBlock> binaryBlocks) {
		return binaryBlocksToFrameObject(binaryBlocks, null);
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a {@code FrameObject}.
	 *
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation of a
	 *            binary-block frame
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(JavaPairRDD<Long, FrameBlock> binaryBlocks,
			FrameMetadata frameMetadata) {

		MatrixCharacteristics mc = (frameMetadata != null) ?
			frameMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
		ValueType[] schema = (frameMetadata != null) ?
			frameMetadata.getFrameSchema().getSchema().toArray(new ValueType[0]) : 
			UtilFunctions.nCopies((int)mc.getCols(), ValueType.STRING);
		
		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(),
			new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo), schema);
		frameObject.setRDDHandle(new RDDObject(binaryBlocks));
		return frameObject;
	}

	/**
	 * Convert a {@code DataFrame} to a {@code MatrixObject}.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject dataFrameToMatrixObject(Dataset<Row> dataFrame) {
		return dataFrameToMatrixObject(dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a {@code MatrixObject}.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param matrixMetadata
	 *            the matrix metadata
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code MatrixObject}
	 */
	public static MatrixObject dataFrameToMatrixObject(Dataset<Row> dataFrame,
			MatrixMetadata matrixMetadata) {
		matrixMetadata = (matrixMetadata != null) ? matrixMetadata : new MatrixMetadata();
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlock = dataFrameToMatrixBinaryBlocks(dataFrame, matrixMetadata);
		MatrixObject mo = binaryBlocksToMatrixObject(binaryBlock, matrixMetadata, false);
		// keep lineage of original dataset to allow bypassing binary block
		// conversion if possible
		mo.getRDDHandle().addLineageChild(new DatasetObject(dataFrame,
				isDataFrameWithIDColumn(matrixMetadata), isVectorBasedDataFrame(matrixMetadata)));
		return mo;
	}

	/**
	 * Convert a {@code DataFrame} to a {@code FrameObject}.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @return the {@code DataFrame} matrix converted to a converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject dataFrameToFrameObject(Dataset<Row> dataFrame) {
		return dataFrameToFrameObject(dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a {@code FrameObject}.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code DataFrame} frame converted to a converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject dataFrameToFrameObject(Dataset<Row> dataFrame,
			FrameMetadata frameMetadata) {
		try {
			// setup meta data and java spark context
			if (frameMetadata == null)
				frameMetadata = new FrameMetadata();
			determineFrameFormatIfNeeded(dataFrame, frameMetadata);
			boolean containsID = isDataFrameWithIDColumn(frameMetadata);
			MatrixCharacteristics mc = frameMetadata.asMatrixCharacteristics();
			if (mc == null)
				mc = new MatrixCharacteristics();

			// convert data frame and obtain column names / schema
			// TODO extend frame schema by column names (right now dropped)
			Pair<String[], ValueType[]> ret = new Pair<>();
			JavaPairRDD<Long, FrameBlock> binaryBlock = FrameRDDConverterUtils.dataFrameToBinaryBlock(jsc(), dataFrame,
					mc, containsID, ret);
			frameMetadata.setFrameSchema(new FrameSchema(Arrays.asList(ret.getValue())));
			frameMetadata.setMatrixCharacteristics(mc); // required due to meta
														// data copy

			return MLContextConversionUtil.binaryBlocksToFrameObject(binaryBlock, frameMetadata);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception converting DataFrame to FrameObject", e);
		}
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToMatrixBinaryBlocks(Dataset<Row> dataFrame) {
		return dataFrameToMatrixBinaryBlocks(dataFrame, null);
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToMatrixBinaryBlocks(Dataset<Row> dataFrame,
			MatrixMetadata matrixMetadata) {
		// handle meta data
		determineMatrixFormatIfNeeded(dataFrame, matrixMetadata);
		MatrixCharacteristics mc = (matrixMetadata != null && matrixMetadata.asMatrixCharacteristics() != null)
				? matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
		boolean containsID = isDataFrameWithIDColumn(matrixMetadata);
		boolean isVector = isVectorBasedDataFrame(matrixMetadata);

		// convert data frame to binary block matrix
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.dataFrameToBinaryBlock(jsc(), dataFrame, mc,
				containsID, isVector);

		// update determined matrix characteristics
		if (matrixMetadata != null)
			matrixMetadata.setMatrixCharacteristics(mc);

		return out;
	}

	/**
	 * Convert a {@code DataFrame} to a {@code JavaPairRDD<Long, FrameBlock>}
	 * binary-block frame.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code DataFrame} matrix converted to a
	 *         {@code JavaPairRDD<Long,
	 *         FrameBlock>} binary-block frame
	 */
	public static JavaPairRDD<Long, FrameBlock> dataFrameToFrameBinaryBlocks(Dataset<Row> dataFrame,
			FrameMetadata frameMetadata) {
		throw new MLContextException("dataFrameToFrameBinaryBlocks is unimplemented");
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
	public static void determineMatrixFormatIfNeeded(Dataset<Row> dataFrame, MatrixMetadata matrixMetadata) {
		if (matrixMetadata == null) {
			return;
		}
		MatrixFormat matrixFormat = matrixMetadata.getMatrixFormat();
		if (matrixFormat != null) {
			return;
		}
		StructType schema = dataFrame.schema();
		boolean hasID = false;
		try {
			schema.fieldIndex(RDDConverterUtils.DF_ID_COLUMN);
			hasID = true;
		} catch (IllegalArgumentException iae) {
		}

		StructField[] fields = schema.fields();
		MatrixFormat mf = null;
		if (hasID) {
			if (fields[1].dataType() instanceof VectorUDT) {
				mf = MatrixFormat.DF_VECTOR_WITH_INDEX;
			} else {
				mf = MatrixFormat.DF_DOUBLES_WITH_INDEX;
			}
		} else {
			if (fields[0].dataType() instanceof VectorUDT) {
				mf = MatrixFormat.DF_VECTOR;
			} else {
				mf = MatrixFormat.DF_DOUBLES;
			}
		}

		if (mf == null) {
			throw new MLContextException("DataFrame format not recognized as an accepted SystemDS MatrixFormat");
		}
		matrixMetadata.setMatrixFormat(mf);
	}

	/**
	 * If the FrameFormat of the DataFrame has not been explicitly specified,
	 * attempt to determine the proper FrameFormat.
	 *
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param frameMetadata
	 *            the frame metadata, if available
	 */
	public static void determineFrameFormatIfNeeded(Dataset<Row> dataFrame, FrameMetadata frameMetadata) {
		FrameFormat frameFormat = frameMetadata.getFrameFormat();
		if (frameFormat != null) {
			return;
		}

		StructType schema = dataFrame.schema();
		boolean hasID = false;
		try {
			schema.fieldIndex(RDDConverterUtils.DF_ID_COLUMN);
			hasID = true;
		} catch (IllegalArgumentException iae) {
		}

		FrameFormat ff = hasID ? FrameFormat.DF_WITH_INDEX : FrameFormat.DF;
		frameMetadata.setFrameFormat(ff);
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
		return (matrixMetadata != null && matrixMetadata.getMatrixFormat() != null
				&& matrixMetadata.getMatrixFormat().hasIDColumn());
	}

	/**
	 * Return whether or not the DataFrame has an ID column.
	 *
	 * @param frameMetadata
	 *            the frame metadata
	 * @return {@code true} if the DataFrame has an ID column, {@code false}
	 *         otherwise.
	 */
	public static boolean isDataFrameWithIDColumn(FrameMetadata frameMetadata) {
		return (frameMetadata != null && frameMetadata.getFrameFormat() != null
				&& frameMetadata.getFrameFormat().hasIDColumn());
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
		return (matrixMetadata != null && matrixMetadata.getMatrixFormat() != null
				&& matrixMetadata.getMatrixFormat().isVectorBased());
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code MatrixObject}
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringCSVToMatrixObject(JavaRDD<String> javaRDD) {
		return javaRDDStringCSVToMatrixObject(javaRDD, null);
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code MatrixObject}
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringCSVToMatrixObject(JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		DataCharacteristics mc = (matrixMetadata != null) ? matrixMetadata.asMatrixCharacteristics()
				: new MatrixCharacteristics();

		MatrixObject matrixObject = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
		JavaPairRDD<LongWritable, Text> javaPairRDD2 = javaPairRDD.mapToPair(new CopyTextInputFunction());
		matrixObject.setRDDHandle(new RDDObject(javaPairRDD2));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code FrameObject}
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringCSVToFrameObject(JavaRDD<String> javaRDD) {
		return javaRDDStringCSVToFrameObject(javaRDD, null);
	}

	/**
	 * Convert a {@code JavaRDD<String>} in CSV format to a {@code FrameObject}
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringCSVToFrameObject(JavaRDD<String> javaRDD,
			FrameMetadata frameMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics mc = (frameMetadata != null) ? frameMetadata.asMatrixCharacteristics()
				: new MatrixCharacteristics();
		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());

		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo),
				frameMetadata.getFrameSchema().getSchema().toArray(new ValueType[0]));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			rdd = FrameRDDConverterUtils.csvToBinaryBlock(jsc(), javaPairRDDText, mc, frameObject.getSchema(), false,
					",", false, -1);
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			return null;
		}
		frameObject.setRDDHandle(new RDDObject(rdd));
		return frameObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in IJV format to a {@code MatrixObject}
	 * . Note that metadata is required for IJV format.
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject javaRDDStringIJVToMatrixObject(JavaRDD<String> javaRDD,
			MatrixMetadata matrixMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics mc = (matrixMetadata != null) ? matrixMetadata.asMatrixCharacteristics()
				: new MatrixCharacteristics();

		MatrixObject matrixObject = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		JavaPairRDD<LongWritable, Text> javaPairRDD2 = javaPairRDD.mapToPair(new CopyTextInputFunction());
		matrixObject.setRDDHandle(new RDDObject(javaPairRDD2));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaRDD<String>} in IJV format to a {@code FrameObject}
	 * . Note that metadata is required for IJV format.
	 *
	 * @param javaRDD
	 *            the Java RDD of strings
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringIJVToFrameObject(JavaRDD<String> javaRDD,
			FrameMetadata frameMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics mc = (frameMetadata != null) ? frameMetadata.asMatrixCharacteristics()
				: new MatrixCharacteristics();

		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());

		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo),
				frameMetadata.getFrameSchema().getSchema().toArray(new ValueType[0]));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			ValueType[] lschema = null;
			if (lschema == null)
				lschema = UtilFunctions.nCopies((int) mc.getCols(), ValueType.STRING);
			rdd = FrameRDDConverterUtils.textCellToBinaryBlock(jsc(), javaPairRDDText, mc, lschema);
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			return null;
		}
		frameObject.setRDDHandle(new RDDObject(rdd));
		return frameObject;
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code MatrixObject}
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringCSVToMatrixObject(RDD<String> rdd) {
		return rddStringCSVToMatrixObject(rdd, null);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code MatrixObject}
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringCSVToMatrixObject(RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringCSVToMatrixObject(javaRDD, matrixMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code FrameObject}
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringCSVToFrameObject(RDD<String> rdd) {
		return rddStringCSVToFrameObject(rdd, null);
	}

	/**
	 * Convert a {@code RDD<String>} in CSV format to a {@code FrameObject}
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringCSVToFrameObject(RDD<String> rdd,
			FrameMetadata frameMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringCSVToFrameObject(javaRDD, frameMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in IJV format to a {@code MatrixObject}.
	 * Note that metadata is required for IJV format.
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @param matrixMetadata
	 *            matrix metadata
	 * @return the {@code RDD<String>} converted to a {@code MatrixObject}
	 */
	public static MatrixObject rddStringIJVToMatrixObject(RDD<String> rdd,
			MatrixMetadata matrixMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringIJVToMatrixObject(javaRDD, matrixMetadata);
	}

	/**
	 * Convert a {@code RDD<String>} in IJV format to a {@code FrameObject}.
	 * Note that metadata is required for IJV format.
	 *
	 * @param rdd
	 *            the RDD of strings
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringIJVToFrameObject(RDD<String> rdd,
			FrameMetadata frameMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringIJVToFrameObject(javaRDD, frameMetadata);
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

		return jsc().parallelize(list);
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaRDD<String>} in CSV format.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param delimiter
	 *            the delimiter
	 * @return the {@code FrameObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> frameObjectToJavaRDDStringCSV(FrameObject frameObject, String delimiter) {
		List<String> list = frameObjectToListStringCSV(frameObject, delimiter);

		return jsc().parallelize(list);
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

		return jsc().parallelize(list);
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaRDD<String>} in IJV format.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> frameObjectToJavaRDDStringIJV(FrameObject frameObject) {
		List<String> list = frameObjectToListStringIJV(frameObject);

		return jsc().parallelize(list);
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

		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc().parallelize(JavaConversions.asScalaBuffer(list), sc().defaultParallelism(), tag);
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

		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc().parallelize(JavaConversions.asScalaBuffer(list), sc().defaultParallelism(), tag);
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

		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc().parallelize(JavaConversions.asScalaBuffer(list), sc().defaultParallelism(), tag);
	}

	/**
	 * Convert a {@code FrameObject} to a {@code RDD<String>} in CSV format.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param delimiter
	 *            the delimiter
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

		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc().parallelize(JavaConversions.asScalaBuffer(list), sc().defaultParallelism(), tag);
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code List<String>} in CSV format.
	 *
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code List<String>}
	 */
	public static List<String> matrixObjectToListStringCSV(MatrixObject matrixObject) {
		MatrixBlock mb = matrixObject.acquireRead();

		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		List<String> list = new ArrayList<>();

		if ( !mb.isEmptyBlock(false) ) {
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
	}

	/**
	 * Convert a {@code FrameObject} to a {@code List<String>} in CSV format.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param delimiter
	 *            the delimiter
	 * @return the {@code FrameObject} converted to a {@code List<String>}
	 */
	public static List<String> frameObjectToListStringCSV(FrameObject frameObject, String delimiter) {
		FrameBlock fb = frameObject.acquireRead();

		int rows = fb.getNumRows();
		int cols = fb.getNumColumns();
		List<String> list = new ArrayList<>();

		for (int i = 0; i < rows; i++) {
			StringBuilder sb = new StringBuilder();
			for (int j = 0; j < cols; j++) {
				if (j > 0) {
					sb.append(delimiter);
				}
				if (fb.get(i, j) != null) {
					sb.append(fb.get(i, j));
				}
			}
			list.add(sb.toString());
		}

		frameObject.release();
		return list;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code List<String>} in IJV format.
	 *
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code List<String>}
	 */
	public static List<String> matrixObjectToListStringIJV(MatrixObject matrixObject) {
		MatrixBlock mb = matrixObject.acquireRead();

		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		List<String> list = new ArrayList<>();

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
	}

	/**
	 * Convert a {@code FrameObject} to a {@code List<String>} in IJV format.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code List<String>}
	 */
	public static List<String> frameObjectToListStringIJV(FrameObject frameObject) {
		FrameBlock fb = frameObject.acquireRead();

		int rows = fb.getNumRows();
		int cols = fb.getNumColumns();
		List<String> list = new ArrayList<>();

		StringBuilder sb = null;
		for (int i = 0; i < rows; i++) {
			sb = new StringBuilder();
			for (int j = 0; j < cols; j++) {
				if (fb.get(i, j) != null) {
					sb = new StringBuilder();
					sb.append(i + 1);
					sb.append(" ");
					sb.append(j + 1);
					sb.append(" ");
					sb.append(fb.get(i, j));
					list.add(sb.toString());
				}
			}
		}

		frameObject.release();
		return list;
	}

	/**
	 * Convert a {@code MatrixObject} to a two-dimensional double array.
	 *
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @return the {@code MatrixObject} converted to a {@code double[][]}
	 */
	public static double[][] matrixObjectTo2DDoubleArray(MatrixObject matrixObject) {
		MatrixBlock mb = matrixObject.acquireRead();
		double[][] matrix = DataConverter.convertToDoubleMatrix(mb);
		matrixObject.release();
		return matrix;
	}

	/**
	 * Convert a {@code MatrixObject} to a {@code DataFrame}.
	 *
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @param isVectorDF
	 *            is the DataFrame a vector DataFrame?
	 * @return the {@code MatrixObject} converted to a {@code DataFrame}
	 */
	public static Dataset<Row> matrixObjectToDataFrame(MatrixObject matrixObject,
			SparkExecutionContext sparkExecutionContext, boolean isVectorDF) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sparkExecutionContext
					.getRDDHandleForMatrixObject(matrixObject, InputInfo.BinaryBlockInputInfo);
			DataCharacteristics mc = matrixObject.getDataCharacteristics();

			return RDDConverterUtils.binaryBlockToDataFrame(spark(), binaryBlocks, mc, isVectorDF);
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
	public static Dataset<Row> frameObjectToDataFrame(FrameObject frameObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<Long, FrameBlock> binaryBlockFrame = (JavaPairRDD<Long, FrameBlock>) sparkExecutionContext
					.getRDDHandleForFrameObject(frameObject, InputInfo.BinaryBlockInputInfo);
			DataCharacteristics mc = frameObject.getDataCharacteristics();

			return FrameRDDConverterUtils.binaryBlockToDataFrame(spark(), binaryBlockFrame, mc,
					frameObject.getSchema());
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting frame object to DataFrame", e);
		}
	}

	/**
	 * Convert a {@code MatrixObject} to a
	 * {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}.
	 *
	 * @param matrixObject
	 *            the {@code MatrixObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code MatrixObject} converted to a
	 *         {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> matrixObjectToBinaryBlocks(MatrixObject matrixObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sparkExecutionContext
					.getRDDHandleForMatrixObject(matrixObject, InputInfo.BinaryBlockInputInfo);
			return binaryBlocks;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting matrix object to binary blocks", e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaPairRDD<Long, FrameBlock>}.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code FrameObject} converted to a
	 *         {@code JavaPairRDD<Long, FrameBlock>}
	 */
	public static JavaPairRDD<Long, FrameBlock> frameObjectToBinaryBlocks(FrameObject frameObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<Long, FrameBlock> binaryBlocks = (JavaPairRDD<Long, FrameBlock>) sparkExecutionContext
					.getRDDHandleForFrameObject(frameObject, InputInfo.BinaryBlockInputInfo);
			return binaryBlocks;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting frame object to binary blocks", e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a two-dimensional string array.
	 *
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code String[][]}
	 */
	public static String[][] frameObjectTo2DStringArray(FrameObject frameObject) {
		FrameBlock fb = frameObject.acquireRead();
		String[][] frame = DataConverter.convertToStringFrame(fb);
		frameObject.release();
		return frame;
	}

	/**
	 * Obtain JavaSparkContext from MLContextProxy.
	 *
	 * @return the Java Spark Context
	 */
	public static JavaSparkContext jsc() {
		return MLContextUtil.getJavaSparkContextFromProxy();
	}

	/**
	 * Obtain SparkContext from MLContextProxy.
	 *
	 * @return the Spark Context
	 */
	public static SparkContext sc() {
		return MLContextUtil.getSparkContextFromProxy();
	}

	/**
	 * Obtain SparkSession from MLContextProxy.
	 *
	 * @return the Spark Session
	 */
	public static SparkSession spark() {
		return MLContextUtil.getSparkSessionFromProxy();
	}
}
