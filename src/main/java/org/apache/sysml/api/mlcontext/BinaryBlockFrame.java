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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * BinaryBlockFrame stores data as a SystemML binary-block frame representation.
 *
 */
public class BinaryBlockFrame {

	JavaPairRDD<Long, FrameBlock> binaryBlocks;
	FrameMetadata frameMetadata;

	/**
	 * Convert a Spark DataFrame to a SystemML binary-block representation.
	 * 
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param frameMetadata
	 *            frame metadata, such as number of rows and columns
	 */
	public BinaryBlockFrame(DataFrame dataFrame, FrameMetadata frameMetadata) {
		this.frameMetadata = frameMetadata;
		binaryBlocks = MLContextConversionUtil.dataFrameToFrameBinaryBlocks(dataFrame, frameMetadata);
	}

	/**
	 * Convert a Spark DataFrame to a SystemML binary-block representation,
	 * specifying the number of rows and columns.
	 * 
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param numRows
	 *            the number of rows
	 * @param numCols
	 *            the number of columns
	 */
	public BinaryBlockFrame(DataFrame dataFrame, long numRows, long numCols) {
		this(dataFrame, new FrameMetadata(numRows, numCols, 
				ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize()));
	}

	/**
	 * Convert a Spark DataFrame to a SystemML binary-block representation.
	 * 
	 * @param dataFrame
	 *            the Spark DataFrame
	 */
	public BinaryBlockFrame(DataFrame dataFrame) {
		this(dataFrame, new FrameMetadata());
	}

	/**
	 * Create a BinaryBlockFrame, specifying the SystemML binary-block frame and
	 * its metadata.
	 * 
	 * @param binaryBlocks
	 *            the {@code JavaPairRDD<Long, FrameBlock>} frame
	 * @param matrixCharacteristics
	 *            the frame metadata as {@code MatrixCharacteristics}
	 */
	public BinaryBlockFrame(JavaPairRDD<Long, FrameBlock> binaryBlocks, MatrixCharacteristics matrixCharacteristics) {
		this.binaryBlocks = binaryBlocks;
		this.frameMetadata = new FrameMetadata(matrixCharacteristics);
	}

	/**
	 * Create a BinaryBlockFrame, specifying the SystemML binary-block frame and
	 * its metadata.
	 * 
	 * @param binaryBlocks
	 *            the {@code JavaPairRDD<Long, FrameBlock>} frame
	 * @param frameMetadata
	 *            the frame metadata as {@code FrameMetadata}
	 */
	public BinaryBlockFrame(JavaPairRDD<Long, FrameBlock> binaryBlocks, FrameMetadata frameMetadata) {
		this.binaryBlocks = binaryBlocks;
		this.frameMetadata = frameMetadata;
	}

	/**
	 * Obtain a SystemML binary-block frame as a
	 * {@code JavaPairRDD<Long, FrameBlock>}
	 * 
	 * @return the SystemML binary-block frame
	 */
	public JavaPairRDD<Long, FrameBlock> getBinaryBlocks() {
		return binaryBlocks;
	}

	/**
	 * Obtain a SystemML binary-block frame as a {@code FrameBlock}
	 * 
	 * @return the SystemML binary-block frame as a {@code FrameBlock}
	 */
	public FrameBlock getFrameBlock() {
		try {
			MatrixCharacteristics mc = getMatrixCharacteristics();
			FrameSchema frameSchema = frameMetadata.getFrameSchema();
			return SparkExecutionContext.toFrameBlock(binaryBlocks, frameSchema.getSchema(),
					(int) mc.getRows(), (int) mc.getCols());
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception while getting FrameBlock from binary-block frame", e);
		}
	}

	/**
	 * Obtain the SystemML binary-block frame characteristics
	 * 
	 * @return the frame metadata as {@code MatrixCharacteristics}
	 */
	public MatrixCharacteristics getMatrixCharacteristics() {
		return frameMetadata.asMatrixCharacteristics();
	}

	/**
	 * Obtain the SystemML binary-block frame metadata
	 * 
	 * @return the frame metadata as {@code FrameMetadata}
	 */
	public FrameMetadata getFrameMetadata() {
		return frameMetadata;
	}

	/**
	 * Set the SystemML binary-block frame metadata
	 * 
	 * @param frameMetadata
	 *            the frame metadata
	 */
	public void setFrameMetadata(FrameMetadata frameMetadata) {
		this.frameMetadata = frameMetadata;
	}

	/**
	 * Set the SystemML binary-block frame as a
	 * {@code JavaPairRDD<Long, FrameBlock>}
	 * 
	 * @param binaryBlocks
	 *            the SystemML binary-block frame
	 */
	public void setBinaryBlocks(JavaPairRDD<Long, FrameBlock> binaryBlocks) {
		this.binaryBlocks = binaryBlocks;
	}

	@Override
	public String toString() {
		if (frameMetadata != null) {
			return frameMetadata.toString();
		} else {
			return super.toString();
		}
	}
}
