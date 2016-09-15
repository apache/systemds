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

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;

/**
 * Matrix encapsulates a SystemML matrix. It allows for easy conversion to
 * various other formats, such as RDDs, JavaRDDs, DataFrames,
 * BinaryBlockMatrices, and double[][]s. After script execution, it offers a
 * convenient format for obtaining SystemML matrix data in Scala tuples.
 *
 */
public class Matrix {

	private MatrixObject matrixObject;
	private SparkExecutionContext sparkExecutionContext;

	public Matrix(MatrixObject matrixObject, SparkExecutionContext sparkExecutionContext) {
		this.matrixObject = matrixObject;
		this.sparkExecutionContext = sparkExecutionContext;
	}

	/**
	 * Obtain the matrix as a SystemML MatrixObject.
	 * 
	 * @return the matrix as a SystemML MatrixObject
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
	public DataFrame toDF() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of doubles with an ID column
	 * 
	 * @return the matrix as a {@code DataFrame} of doubles with an ID column
	 */
	public DataFrame toDFDoubleWithIDColumn() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of doubles with no ID column
	 * 
	 * @return the matrix as a {@code DataFrame} of doubles with no ID column
	 */
	public DataFrame toDFDoubleNoIDColumn() {
		DataFrame df = MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, false);
		return df.drop(RDDConverterUtils.DF_ID_COLUMN);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of vectors with an ID column
	 * 
	 * @return the matrix as a {@code DataFrame} of vectors with an ID column
	 */
	public DataFrame toDFVectorWithIDColumn() {
		return MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, true);
	}

	/**
	 * Obtain the matrix as a {@code DataFrame} of vectors with no ID column
	 * 
	 * @return the matrix as a {@code DataFrame} of vectors with no ID column
	 */
	public DataFrame toDFVectorNoIDColumn() {
		DataFrame df = MLContextConversionUtil.matrixObjectToDataFrame(matrixObject, sparkExecutionContext, true);
		return df.drop(RDDConverterUtils.DF_ID_COLUMN);
	}

	/**
	 * Obtain the matrix as a {@code BinaryBlockMatrix}
	 * 
	 * @return the matrix as a {@code BinaryBlockMatrix}
	 */
	public BinaryBlockMatrix toBinaryBlockMatrix() {
		return MLContextConversionUtil.matrixObjectToBinaryBlockMatrix(matrixObject, sparkExecutionContext);
	}

	/**
	 * Obtain the matrix metadata
	 * 
	 * @return the matrix metadata
	 */
	public MatrixMetadata getMatrixMetadata() {
		return new MatrixMetadata(matrixObject.getMatrixCharacteristics());
	}

	@Override
	public String toString() {
		return matrixObject.toString();
	}
}
