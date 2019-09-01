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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;

/**
 * Matrix encapsulates a SystemDS matrix. It allows for easy conversion to
 * various other formats, such as RDDs, JavaRDDs, DataFrames, and double[][]s.
 * After script execution, it offers a convenient format for obtaining SystemDS
 * matrix data in Scala tuples.
 *
 */
public class Matrix {

	private MatrixObject matrixObject;
	private SparkExecutionContext sparkExecutionContext;
	private JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks;
	private MatrixMetadata matrixMetadata;

	public Matrix(MatrixObject matrixObject, SparkExecutionContext sparkExecutionContext) {
		this.matrixObject = matrixObject;
		this.sparkExecutionContext = sparkExecutionContext;
		this.matrixMetadata = new MatrixMetadata((MatrixCharacteristics) matrixObject.getDataCharacteristics());
	}

	/**
	 * Convert a Spark DataFrame to a SystemDS binary-block representation.
	 *
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param matrixMetadata
	 *            matrix metadata, such as number of rows and columns
	 */
	public Matrix(Dataset<Row> dataFrame, MatrixMetadata matrixMetadata) {
		this.matrixMetadata = matrixMetadata;
		binaryBlocks = MLContextConversionUtil.dataFrameToMatrixBinaryBlocks(dataFrame, matrixMetadata);
	}

	/**
	 * Convert a Spark DataFrame to a SystemDS binary-block representation,
	 * specifying the number of rows and columns.
	 *
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param numRows
	 *            the number of rows
	 * @param numCols
	 *            the number of columns
	 */
	public Matrix(Dataset<Row> dataFrame, long numRows, long numCols) {
		this(dataFrame, new MatrixMetadata(numRows, numCols, ConfigurationManager.getBlocksize()));
	}

	/**
	 * Create a Matrix, specifying the SystemDS binary-block matrix and its
	 * metadata.
	 *
	 * @param binaryBlocks
	 *            the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 * @param matrixMetadata
	 *            matrix metadata, such as number of rows and columns
	 */
	public Matrix(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks, MatrixMetadata matrixMetadata) {
		this.binaryBlocks = binaryBlocks;
		this.matrixMetadata = matrixMetadata;
	}

	/**
	 * Convert a Spark DataFrame to a SystemDS binary-block representation.
	 *
	 * @param dataFrame
	 *            the Spark DataFrame
	 */
	public Matrix(Dataset<Row> dataFrame) {
		this(dataFrame, new MatrixMetadata());
	}

	/**
	 * Obtain the matrix as a SystemDS MatrixObject.
	 *
	 * @return the matrix as a SystemDS MatrixObject
	 */
	public MatrixObject toMatrixObject() {
		return matrixObject;
	}

	/**
	 * Obtain the matrix as a two-dimensional double array
	 *
	 * @return the matrix as a two-dimensional double array
	 */
	public double[][] to2DDoubleArray() {
		return MLContextConversionUtil.matrixObjectTo2DDoubleArray(matrixObject);
	}

	/**
	 * Obtain the matrix as a {@code JavaRDD<String>} in IJV format
	 *
	 * @return the matrix as a {@code JavaRDD<String>} in IJV format
	 */
	public JavaRDD<String> toJavaRDDStringIJV() {
		return MLContextConversionUtil.matrixObjectToJavaRDDStringIJV(matrixObject);
	}

	/**
	 * Obtain the matrix as a {@code JavaRDD<String>} in CSV format
	 *
	 * @return the matrix as a {@code JavaRDD<String>} in CSV format
	 */
	public JavaRDD<String> toJavaRDDStringCSV() {
		return MLContextConversionUtil.matrixObjectToJavaRDDStringCSV(matrixObject);
	}

	/**
	 * Obtain the matrix as a {@code RDD<String>} in CSV format
	 *
	 * @return the matrix as a {@code RDD<String>} in CSV format
	 */
	public RDD<String> toRDDStringCSV() {
		return MLContextConversionUtil.matrixObjectToRDDStringCSV(matrixObject);
	}

	/**
	 * Obtain the matrix as a {@code RDD<String>} in IJV format
	 *
	 * @return the matrix as a {@code RDD<String>} in IJV format
	 */
	public RDD<String> toRDDStringIJV() {
		return MLContextConversionUtil.matrixObjectToRDDStringIJV(matrixObject);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of doubles with an ID column
	 *
	 * @return the matrix as a {@code DataFrame} of doubles with an ID column
	 */
	public Dataset<Row> toDF() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of doubles with an ID column
	 *
	 * @return the matrix as a {@code DataFrame} of doubles with an ID column
	 */
	public Dataset<Row> toDFDoubleWithIDColumn() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of doubles with no ID column
	 *
	 * @return the matrix as a {@code DataFrame} of doubles with no ID column
	 */
	public Dataset<Row> toDFDoubleNoIDColumn() {
		Dataset<Row> df = MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
		return df.drop(RDDConverterUtils.DF_ID_COLUMN);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of vectors with an ID column
	 *
	 * @return the matrix as a {@code DataFrame} of vectors with an ID column
	 */
	public Dataset<Row> toDFVectorWithIDColumn() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, true);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of vectors with no ID column
	 *
	 * @return the matrix as a {@code DataFrame} of vectors with no ID column
	 */
	public Dataset<Row> toDFVectorNoIDColumn() {
		Dataset<Row> df = MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, true);
		return df.drop(RDDConverterUtils.DF_ID_COLUMN);
	}

	/**
	 * Obtain the matrix as a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 *
	 * @return the matrix as a {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public JavaPairRDD<MatrixIndexes, MatrixBlock> toBinaryBlocks() {
		if (binaryBlocks != null) {
			return binaryBlocks;
		} else if (matrixObject != null) {
			binaryBlocks = MLContextConversionUtil.matrixObjectToBinaryBlocks(matrixObject, sparkExecutionContext);
			DataCharacteristics mc = matrixObject.getDataCharacteristics();
			matrixMetadata = new MatrixMetadata(mc);
			return binaryBlocks;
		}
		throw new MLContextException("No binary blocks or MatrixObject found");
	}

	/**
	 * Obtain the matrix as a {@code MatrixBlock}
	 *
	 * @return the matrix as a {@code MatrixBlock}
	 */
	public MatrixBlock toMatrixBlock() {
		if (matrixMetadata == null) {
			throw new MLContextException("Matrix metadata required to convert binary blocks to a MatrixBlock.");
		}
		if (binaryBlocks != null) {
			return MLContextConversionUtil.binaryBlocksToMatrixBlock(binaryBlocks, matrixMetadata);
		} else if (matrixObject != null) {
			return MLContextConversionUtil.binaryBlocksToMatrixBlock(toBinaryBlocks(), matrixMetadata);
		}
		throw new MLContextException("No binary blocks or MatrixObject found");
	}

	/**
	 * Obtain the matrix metadata
	 *
	 * @return the matrix metadata
	 */
	public MatrixMetadata getMatrixMetadata() {
		return matrixMetadata;
	}

	/**
	 * If {@code MatrixObject} is available, output
	 * {@code MatrixObject.toString()}. If {@code MatrixObject} is not available
	 * but {@code MatrixMetadata} is available, output
	 * {@code MatrixMetadata.toString()}. Otherwise output
	 * {@code Object.toString()}.
	 */
	@Override
	public String toString() {
		if (matrixObject != null) {
			return matrixObject.toString();
		} else if (matrixMetadata != null) {
			return matrixMetadata.toString();
		} else {
			return super.toString();
		}
	}

	/**
	 * Whether or not this matrix contains data as binary blocks
	 *
	 * @return {@code true} if data as binary blocks are present, {@code false}
	 *         otherwise.
	 */
	public boolean hasBinaryBlocks() {
		return (binaryBlocks != null);
	}

	/**
	 * Whether or not this matrix contains data as a MatrixObject
	 *
	 * @return {@code true} if data as binary blocks are present, {@code false}
	 *         otherwise.
	 */
	public boolean hasMatrixObject() {
		return (matrixObject != null);
	}
}
