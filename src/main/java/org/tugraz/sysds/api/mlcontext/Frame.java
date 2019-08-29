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
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

/**
 * Frame encapsulates a SystemDS frame.
 *
 */
public class Frame {

	private FrameObject frameObject;
	private SparkExecutionContext sparkExecutionContext;
	private JavaPairRDD<Long, FrameBlock> binaryBlocks;
	private FrameMetadata frameMetadata;

	public Frame(FrameObject frameObject, SparkExecutionContext sparkExecutionContext) {
		this.frameObject = frameObject;
		this.sparkExecutionContext = sparkExecutionContext;
		this.frameMetadata = new FrameMetadata(frameObject.getDataCharacteristics());
	}

	/**
	 * Convert a Spark DataFrame to a SystemDS binary-block representation.
	 *
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param frameMetadata
	 *            frame metadata, such as number of rows and columns
	 */
	public Frame(Dataset<Row> dataFrame, FrameMetadata frameMetadata) {
		this.frameMetadata = frameMetadata;
		binaryBlocks = MLContextConversionUtil.dataFrameToFrameBinaryBlocks(dataFrame, frameMetadata);
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
	public Frame(Dataset<Row> dataFrame, long numRows, long numCols) {
		this(dataFrame, new FrameMetadata(numRows, numCols, ConfigurationManager.getBlocksize(),
				ConfigurationManager.getBlocksize()));
	}

	/**
	 * Convert a Spark DataFrame to a SystemDS binary-block representation.
	 *
	 * @param dataFrame
	 *            the Spark DataFrame
	 */
	public Frame(Dataset<Row> dataFrame) {
		this(dataFrame, new FrameMetadata());
	}

	/**
	 * Create a Frame, specifying the SystemDS binary-block frame and its
	 * metadata.
	 *
	 * @param binaryBlocks
	 *            the {@code JavaPairRDD<Long, FrameBlock>} frame
	 * @param frameMetadata
	 *            frame metadata, such as number of rows and columnss
	 */
	public Frame(JavaPairRDD<Long, FrameBlock> binaryBlocks, FrameMetadata frameMetadata) {
		this.binaryBlocks = binaryBlocks;
		this.frameMetadata = frameMetadata;
	}

	/**
	 * Obtain the frame as a SystemDS FrameObject.
	 *
	 * @return the frame as a SystemDS FrameObject
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
	public Dataset<Row> toDF() {
		return MLContextConversionUtil.frameObjectToDataFrame(frameObject, sparkExecutionContext);
	}

	/**
	 * Obtain the frame as a {@code JavaPairRDD<Long, FrameBlock>}
	 *
	 * @return the frame as a {@code JavaPairRDD<Long, FrameBlock>}
	 */
	public JavaPairRDD<Long, FrameBlock> toBinaryBlocks() {
		if (binaryBlocks != null) {
			return binaryBlocks;
		} else if (frameObject != null) {
			binaryBlocks = MLContextConversionUtil.frameObjectToBinaryBlocks(frameObject, sparkExecutionContext);
			DataCharacteristics mc = frameObject.getDataCharacteristics();
			frameMetadata = new FrameMetadata(mc);
			return binaryBlocks;
		}
		throw new MLContextException("No binary blocks or FrameObject found");
	}

	/**
	 * Obtain the frame metadata
	 *
	 * @return the frame metadata
	 */
	public FrameMetadata getFrameMetadata() {
		return frameMetadata;
	}

	@Override
	public String toString() {
		return frameObject.toString();
	}

	/**
	 * Whether or not this frame contains data as binary blocks
	 *
	 * @return {@code true} if data as binary blocks are present, {@code false}
	 *         otherwise.
	 */
	public boolean hasBinaryBlocks() {
		return (binaryBlocks != null);
	}

	/**
	 * Whether or not this frame contains data as a FrameObject
	 *
	 * @return {@code true} if data as binary blocks are present, {@code false}
	 *         otherwise.
	 */
	public boolean hasFrameObject() {
		return (frameObject != null);
	}
}
