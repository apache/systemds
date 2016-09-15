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
		return MLContextConversionUtil.frameObjectTo2DStringArray(frameObject);
	}

	/**
	 * Obtain the frame as a {@code JavaRDD<String>} in IJV format
	 * 
	 * @return the frame as a {@code JavaRDD<String>} in IJV format
	 */
	public JavaRDD<String> toJavaRDDStringIJV() {
		return MLContextConversionUtil.frameObjectToJavaRDDStringIJV(frameObject);
	}

	/**
	 * Obtain the frame as a {@code JavaRDD<String>} in CSV format
	 * 
	 * @return the frame as a {@code JavaRDD<String>} in CSV format
	 */
	public JavaRDD<String> toJavaRDDStringCSV() {
		return MLContextConversionUtil.frameObjectToJavaRDDStringCSV(frameObject, ",");
	}

	/**
	 * Obtain the frame as a {@code RDD<String>} in CSV format
	 * 
	 * @return the frame as a {@code RDD<String>} in CSV format
	 */
	public RDD<String> toRDDStringCSV() {
		return MLContextConversionUtil.frameObjectToRDDStringCSV(frameObject, ",");
	}

	/**
	 * Obtain the frame as a {@code RDD<String>} in IJV format
	 * 
	 * @return the frame as a {@code RDD<String>} in IJV format
	 */
	public RDD<String> toRDDStringIJV() {
		return MLContextConversionUtil.frameObjectToRDDStringIJV(frameObject);
	}

	/**
	 * Obtain the frame as a {@code DataFrame}
	 * 
	 * @return the frame as a {@code DataFrame}
	 */
	public DataFrame toDF() {
		return MLContextConversionUtil.frameObjectToDataFrame(frameObject, sparkExecutionContext);
	}

	/**
	 * Obtain the matrix as a {@code BinaryBlockFrame}
	 * 
	 * @return the matrix as a {@code BinaryBlockFrame}
	 */
	public BinaryBlockFrame toBinaryBlockFrame() {
		return MLContextConversionUtil.frameObjectToBinaryBlockFrame(frameObject, sparkExecutionContext);
	}

	/**
	 * Obtain the frame metadata
	 * 
	 * @return the frame metadata
	 */
	public FrameMetadata getFrameMetadata() {
		return new FrameMetadata(frameObject.getMatrixCharacteristics());
	}

	@Override
	public String toString() {
		return frameObject.toString();
	}
}
