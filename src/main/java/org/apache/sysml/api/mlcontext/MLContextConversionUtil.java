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
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.MLContextProxy;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
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
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;

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
			MatrixCharacteristics mc = (matrixMetadata != null) ? 
					matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics(matrixBlock.getNumRows(), 
					matrixBlock.getNumColumns(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());

			MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
					new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
			
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
			MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContextForAPI();
			SparkContext sparkContext = activeMLContext.getSparkContext();
			@SuppressWarnings("resource")
			JavaSparkContext javaSparkContext = new JavaSparkContext(sparkContext);
			JavaRDD<String> javaRDD = javaSparkContext.parallelize(lines);
			if ((matrixMetadata == null) || (matrixMetadata.getMatrixFormat() == MatrixFormat.CSV)) {
				return javaRDDStringCSVToMatrixObject(variableName, javaRDD, matrixMetadata);
			} else if (matrixMetadata.getMatrixFormat() == MatrixFormat.IJV) {
				return javaRDDStringIJVToMatrixObject(variableName, javaRDD, matrixMetadata);
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
			MatrixCharacteristics mc = (matrixMetadata != null) ? 
					matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
			MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
					new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
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
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code FrameBlock} converted to a {@code FrameObject}
	 */
	public static FrameObject frameBlockToFrameObject(String variableName, FrameBlock frameBlock,
			FrameMetadata frameMetadata) {
		try {
			MatrixCharacteristics mc = (frameMetadata != null) ? 
					frameMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
			MatrixFormatMetaData mtd = new MatrixFormatMetaData(mc, 
					OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(), mtd);
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

		MatrixCharacteristics mc = (matrixMetadata != null) ?
			matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> javaPairRdd = binaryBlocks.mapToPair(new CopyBlockPairFunction());
		
		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(),
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		matrixObject.setRDDHandle(new RDDObject(javaPairRdd, variableName));
		return matrixObject;
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation of a
	 *            binary-block frame
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(String variableName,
			JavaPairRDD<Long, FrameBlock> binaryBlocks) {
		return binaryBlocksToFrameObject(variableName, binaryBlocks, null);
	}

	/**
	 * Convert a {@code JavaPairRDD<Long, FrameBlock>} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param binaryBlocks
	 *            {@code JavaPairRDD<Long, FrameBlock>} representation of a
	 *            binary-block frame
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code JavaPairRDD<Long, FrameBlock>} frame converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject binaryBlocksToFrameObject(String variableName, JavaPairRDD<Long, FrameBlock> binaryBlocks,
			FrameMetadata frameMetadata) {

		MatrixCharacteristics mc = (frameMetadata != null) ? 
				frameMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();

		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
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
		MatrixMetadata matrixMetadata) 
	{
		matrixMetadata = (matrixMetadata!=null) ? matrixMetadata : new MatrixMetadata();
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlock = dataFrameToMatrixBinaryBlocks(dataFrame, matrixMetadata);
		return binaryBlocksToMatrixObject(variableName, binaryBlock, matrixMetadata);
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
	 */
	public static FrameObject dataFrameToFrameObject(String variableName, DataFrame dataFrame) {
		return dataFrameToFrameObject(variableName, dataFrame, null);
	}

	/**
	 * Convert a {@code DataFrame} to a {@code FrameObject}.
	 * 
	 * @param variableName
	 *            name of the variable associated with the frame
	 * @param dataFrame
	 *            the Spark {@code DataFrame}
	 * @param frameMetadata
	 *            the frame metadata
	 * @return the {@code DataFrame} frame converted to a converted to a
	 *         {@code FrameObject}
	 */
	public static FrameObject dataFrameToFrameObject(String variableName, DataFrame dataFrame,
			FrameMetadata frameMetadata) {
		try {
			if (frameMetadata == null) {
				frameMetadata = new FrameMetadata();
			}

			JavaSparkContext javaSparkContext = MLContextUtil
					.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());
			boolean containsID = isDataFrameWithIDColumn(frameMetadata);
			MatrixCharacteristics matrixCharacteristics = frameMetadata.asMatrixCharacteristics();
			if (matrixCharacteristics == null) {
				matrixCharacteristics = new MatrixCharacteristics();
				long rows = dataFrame.count();
				int cols = dataFrame.columns().length;
				matrixCharacteristics.setDimension(rows, cols);
				frameMetadata.setMatrixCharacteristics(matrixCharacteristics);
			}
			JavaPairRDD<Long, FrameBlock> binaryBlock = FrameRDDConverterUtils.dataFrameToBinaryBlock(javaSparkContext,
					dataFrame, matrixCharacteristics, containsID);

			return MLContextConversionUtil.binaryBlocksToFrameObject(variableName, binaryBlock, frameMetadata);
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToMatrixBinaryBlocks(DataFrame dataFrame) {
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> dataFrameToMatrixBinaryBlocks(
			DataFrame dataFrame, MatrixMetadata matrixMetadata) 
	{
		//handle meta data
		determineMatrixFormatIfNeeded(dataFrame, matrixMetadata);
		MatrixCharacteristics mc = (matrixMetadata != null && matrixMetadata.asMatrixCharacteristics()!=null) ?
				matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
		boolean containsID = isDataFrameWithIDColumn(matrixMetadata);
		boolean isVector = isVectorBasedDataFrame(matrixMetadata);
	
		//get spark context
		JavaSparkContext sc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());

		//convert data frame to binary block matrix
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = RDDConverterUtils
				.dataFrameToBinaryBlock(sc, dataFrame, mc, containsID, isVector);
		
		//update determined matrix characteristics
		if( matrixMetadata != null )
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
	public static JavaPairRDD<Long, FrameBlock> dataFrameToFrameBinaryBlocks(DataFrame dataFrame,
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
	public static void determineMatrixFormatIfNeeded(DataFrame dataFrame, MatrixMetadata matrixMetadata) {
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
		return (matrixMetadata != null 
			&& matrixMetadata.getMatrixFormat() != null
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
		return (frameMetadata != null 
			&& frameMetadata.getFrameFormat() != null
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
		return (matrixMetadata != null 
			&& matrixMetadata.getMatrixFormat() != null
			&& matrixMetadata.getMatrixFormat().isVectorBased());
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
		MatrixCharacteristics mc = (matrixMetadata != null) ? 
				matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();

		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(),
				new MatrixFormatMetaData(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
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
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringCSVToFrameObject(String variableName, JavaRDD<String> javaRDD,
			FrameMetadata frameMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics mc = (frameMetadata != null) ? 
				frameMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();
		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());

		JavaSparkContext jsc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());

		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			rdd = FrameRDDConverterUtils.csvToBinaryBlock(jsc, javaPairRDDText, mc, false, ",", false, -1);
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
		MatrixCharacteristics mc = (matrixMetadata != null) ? 
				matrixMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();

		MatrixObject matrixObject = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
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
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code JavaRDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject javaRDDStringIJVToFrameObject(String variableName, JavaRDD<String> javaRDD,
			FrameMetadata frameMetadata) {
		JavaPairRDD<LongWritable, Text> javaPairRDD = javaRDD.mapToPair(new ConvertStringToLongTextPair());
		MatrixCharacteristics mc = (frameMetadata != null) ? 
				frameMetadata.asMatrixCharacteristics() : new MatrixCharacteristics();

		JavaPairRDD<LongWritable, Text> javaPairRDDText = javaPairRDD.mapToPair(new CopyTextInputFunction());

		JavaSparkContext jsc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());

		FrameObject frameObject = new FrameObject(OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		JavaPairRDD<Long, FrameBlock> rdd;
		try {
			List<ValueType> lschema = null;
			if (lschema == null)
				lschema = Collections.nCopies((int) mc.getCols(), ValueType.STRING);
			rdd = FrameRDDConverterUtils.textCellToBinaryBlock(jsc, javaPairRDDText, mc, lschema);
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
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringCSVToFrameObject(String variableName, RDD<String> rdd,
			FrameMetadata frameMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringCSVToFrameObject(variableName, javaRDD, frameMetadata);
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
	 * @param frameMetadata
	 *            frame metadata
	 * @return the {@code RDD<String>} converted to a {@code FrameObject}
	 */
	public static FrameObject rddStringIJVToFrameObject(String variableName, RDD<String> rdd,
			FrameMetadata frameMetadata) {
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> javaRDD = JavaRDD.fromRDD(rdd, tag);
		return javaRDDStringIJVToFrameObject(variableName, javaRDD, frameMetadata);
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
		MatrixCharacteristics mc = binaryBlockMatrix.getMatrixCharacteristics();
		return RDDConverterUtils.binaryBlockToTextCell(binaryBlock, mc);
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
		return JavaRDD.toRDD(javaRDD);
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

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContextForAPI();
		SparkContext sc = activeMLContext.getSparkContext();
		@SuppressWarnings("resource")
		JavaSparkContext jsc = new JavaSparkContext(sc);
		return jsc.parallelize(list);
	}

	/**
	 * Convert a {@code FrameObject} to a {@code JavaRDD<String>} in CSV format.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @return the {@code FrameObject} converted to a {@code JavaRDD<String>}
	 */
	public static JavaRDD<String> frameObjectToJavaRDDStringCSV(FrameObject frameObject, String delimiter) {
		List<String> list = frameObjectToListStringCSV(frameObject, delimiter);

		JavaSparkContext jsc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());
		return jsc.parallelize(list);
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

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContextForAPI();
		SparkContext sc = activeMLContext.getSparkContext();
		@SuppressWarnings("resource")
		JavaSparkContext jsc = new JavaSparkContext(sc);
		return jsc.parallelize(list);
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

		JavaSparkContext jsc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());
		return jsc.parallelize(list);
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

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContextForAPI();
		SparkContext sc = activeMLContext.getSparkContext();
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
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

		SparkContext sc = MLContextUtil.getSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
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

		MLContext activeMLContext = (MLContext) MLContextProxy.getActiveMLContextForAPI();
		SparkContext sc = activeMLContext.getSparkContext();
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
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

		SparkContext sc = MLContextUtil.getSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());
		ClassTag<String> tag = scala.reflect.ClassTag$.MODULE$.apply(String.class);
		return sc.parallelize(JavaConversions.asScalaBuffer(list), sc.defaultParallelism(), tag);
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
	public static double[][] matrixObjectTo2DDoubleArray(MatrixObject matrixObject) {
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
			MatrixCharacteristics mc = matrixObject.getMatrixCharacteristics();

			SparkContext sc = ((MLContext) MLContextProxy.getActiveMLContextForAPI()).getSparkContext();
			SQLContext sqlctx = new SQLContext(sc);
			
			return RDDConverterUtils.binaryBlockToDataFrame(sqlctx, binaryBlockMatrix, mc, isVectorDF);			
		} 
		catch (DMLRuntimeException e) {
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

			JavaSparkContext jsc = MLContextUtil.getJavaSparkContext((MLContext) MLContextProxy.getActiveMLContextForAPI());

			return FrameRDDConverterUtils.binaryBlockToDataFrame(binaryBlockFrame, matrixCharacteristics, jsc);
		} 
		catch (DMLRuntimeException e) {
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
			return new BinaryBlockMatrix(binaryBlock, matrixCharacteristics);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting matrix object to BinaryBlockMatrix", e);
		}
	}

	/**
	 * Convert a {@code FrameObject} to a {@code BinaryBlockFrame}.
	 * 
	 * @param frameObject
	 *            the {@code FrameObject}
	 * @param sparkExecutionContext
	 *            the Spark execution context
	 * @return the {@code FrameObject} converted to a {@code BinaryBlockFrame}
	 */
	public static BinaryBlockFrame frameObjectToBinaryBlockFrame(FrameObject frameObject,
			SparkExecutionContext sparkExecutionContext) {
		try {
			@SuppressWarnings("unchecked")
			JavaPairRDD<Long, FrameBlock> binaryBlock = (JavaPairRDD<Long, FrameBlock>) sparkExecutionContext
					.getRDDHandleForFrameObject(frameObject, InputInfo.BinaryBlockInputInfo);
			MatrixCharacteristics matrixCharacteristics = frameObject.getMatrixCharacteristics();
			FrameSchema fs = new FrameSchema(frameObject.getSchema());
			FrameMetadata fm = new FrameMetadata(fs, matrixCharacteristics);
			return new BinaryBlockFrame(binaryBlock, fm);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting frame object to BinaryBlockFrame", e);
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
		try {
			FrameBlock fb = frameObject.acquireRead();
			String[][] frame = DataConverter.convertToStringFrame(fb);
			frameObject.release();
			return frame;
		} catch (CacheException e) {
			throw new MLContextException("CacheException while converting frame object to 2D string array", e);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("DMLRuntimeException while converting frame object to 2D string array", e);
		}
	}
}
