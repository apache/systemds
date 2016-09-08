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
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;

/**
 * Frame encapsulates a SystemML frame.
 *
 */
public class Frame {

	private FrameObject frameObject;
	private SparkExecutionContext sparkExecutionContext;

	public Frame(FrameObject frameObject, SparkExecutionContext sparkExecutionContext) {
		this.frameObject = frameObject;
		this.sparkExecutionContext = sparkExecutionContext;
	}

	/**
	 * Obtain the frame as a SystemML FrameObject.
	 * 
	 * @return the frame as a SystemML FrameObject
	 */
	public FrameObject toFrameObject() {
		return frameObject;
	}

	/**
	 * Obtain the frame as a two-dimensional String array
	 * 
	 * @return the frame as a two-dimensional String array
	 */
	public String[][] to2DStringArray() {
		String[][] strArray = MLContextConversionUtil.frameObjectTo2DStringArray(frameObject);
		return strArray;
	}

	/**
	 * Obtain the frame as a {@code JavaRDD<String>} in IJV format
	 * 
	 * @return the frame as a {@code JavaRDD<String>} in IJV format
	 */
	public JavaRDD<String> toJavaRDDStringIJV() {
		JavaRDD<String> javaRDDStringIJV = MLContextConversionUtil.frameObjectToJavaRDDStringIJV(frameObject);
		return javaRDDStringIJV;
	}

	/**
	 * Obtain the frame as a {@code JavaRDD<String>} in CSV format
	 * 
	 * @return the frame as a {@code JavaRDD<String>} in CSV format
	 */
	public JavaRDD<String> toJavaRDDStringCSV() {
		JavaRDD<String> javaRDDStringCSV = MLContextConversionUtil.frameObjectToJavaRDDStringCSV(frameObject, ",");
		return javaRDDStringCSV;
	}

	/**
	 * Obtain the frame as a {@code RDD<String>} in CSV format
	 * 
	 * @return the frame as a {@code RDD<String>} in CSV format
	 */
	public RDD<String> toRDDStringCSV() {
		RDD<String> rddStringCSV = MLContextConversionUtil.frameObjectToRDDStringCSV(frameObject, ",");
		return rddStringCSV;
	}

	/**
	 * Obtain the frame as a {@code RDD<String>} in IJV format
	 * 
	 * @return the frame as a {@code RDD<String>} in IJV format
	 */
	public RDD<String> toRDDStringIJV() {
		RDD<String> rddStringIJV = MLContextConversionUtil.frameObjectToRDDStringIJV(frameObject);
		return rddStringIJV;
	}

	/**
	 * Obtain the frame as a {@code DataFrame}
	 * 
	 * @return the frame as a {@code DataFrame}
	 */
	public DataFrame toDF() {
		DataFrame df = MLContextConversionUtil.frameObjectToDataFrame(frameObject, sparkExecutionContext);
		return df;
	}

	/**
	 * Obtain the matrix as a {@code BinaryBlockFrame}
	 * 
	 * @return the matrix as a {@code BinaryBlockFrame}
	 */
	public BinaryBlockFrame toBinaryBlockFrame() {
		BinaryBlockFrame binaryBlockFrame = MLContextConversionUtil.frameObjectToBinaryBlockFrame(frameObject,
				sparkExecutionContext);
		return binaryBlockFrame;
	}

	/**
	 * Obtain the frame metadata
	 * 
	 * @return the frame metadata
	 */
	public FrameMetadata getFrameMetadata() {
		MatrixCharacteristics matrixCharacteristics = frameObject.getMatrixCharacteristics();
		FrameMetadata frameMetadata = new FrameMetadata(matrixCharacteristics);
		return frameMetadata;
	}

	@Override
	public String toString() {
		return frameObject.toString();
	}
}
