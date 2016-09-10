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
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * BinaryBlockMatrix stores data as a SystemML binary-block matrix representation.
 *
 */
public class BinaryBlockMatrix {

	JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks;
	MatrixMetadata matrixMetadata;

	/**
	 * Convert a Spark DataFrame to a SystemML binary-block representation.
	 * 
	 * @param dataFrame
	 *            the Spark DataFrame
	 * @param matrixMetadata
	 *            matrix metadata, such as number of rows and columns
	 */
	public BinaryBlockMatrix(DataFrame dataFrame, MatrixMetadata matrixMetadata) {
		this.matrixMetadata = matrixMetadata;
		binaryBlocks = MLContextConversionUtil.dataFrameToMatrixBinaryBlocks(dataFrame, matrixMetadata);
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
	public BinaryBlockMatrix(DataFrame dataFrame, long numRows, long numCols) {
		this(dataFrame, new MatrixMetadata(numRows, numCols, 
				ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize()));
	}

	/**
	 * Convert a Spark DataFrame to a SystemML binary-block representation.
	 * 
	 * @param dataFrame
	 *            the Spark DataFrame
	 */
	public BinaryBlockMatrix(DataFrame dataFrame) {
		this(dataFrame, new MatrixMetadata());
	}

	/**
	 * Create a BinaryBlockMatrix, specifying the SystemML binary-block matrix
	 * and its metadata.
	 * 
	 * @param binaryBlocks
	 *            the {@code JavaPairRDD<MatrixIndexes, MatrixBlock>} matrix
	 * @param matrixCharacteristics
	 *            the matrix metadata as {@code MatrixCharacteristics}
	 */
	public BinaryBlockMatrix(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks,
			MatrixCharacteristics matrixCharacteristics) {
		this.binaryBlocks = binaryBlocks;
		this.matrixMetadata = new MatrixMetadata(matrixCharacteristics);
	}

	/**
	 * Obtain a SystemML binary-block matrix as a
	 * {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * 
	 * @return the SystemML binary-block matrix
	 */
	public JavaPairRDD<MatrixIndexes, MatrixBlock> getBinaryBlocks() {
		return binaryBlocks;
	}

	/**
	 * Obtain a SystemML binary-block matrix as a {@code MatrixBlock}
	 * 
	 * @return the SystemML binary-block matrix as a {@code MatrixBlock}
	 */
	public MatrixBlock getMatrixBlock() {
		try {
			MatrixCharacteristics mc = getMatrixCharacteristics();
			MatrixBlock mb = SparkExecutionContext.toMatrixBlock(binaryBlocks, (int) mc.getRows(), (int) mc.getCols(),
					mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros());
			return mb;
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception while getting MatrixBlock from binary-block matrix", e);
		}
	}

	/**
	 * Obtain the SystemML binary-block matrix characteristics
	 * 
	 * @return the matrix metadata as {@code MatrixCharacteristics}
	 */
	public MatrixCharacteristics getMatrixCharacteristics() {
		return matrixMetadata.asMatrixCharacteristics();
	}

	/**
	 * Obtain the SystemML binary-block matrix metadata
	 * 
	 * @return the matrix metadata as {@code MatrixMetadata}
	 */
	public MatrixMetadata getMatrixMetadata() {
		return matrixMetadata;
	}

	/**
	 * Set the SystemML binary-block matrix metadata
	 * 
	 * @param matrixMetadata
	 *            the matrix metadata
	 */
	public void setMatrixMetadata(MatrixMetadata matrixMetadata) {
		this.matrixMetadata = matrixMetadata;
	}

	/**
	 * Set the SystemML binary-block matrix as a
	 * {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * 
	 * @param binaryBlocks
	 *            the SystemML binary-block matrix
	 */
	public void setBinaryBlocks(JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks) {
		this.binaryBlocks = binaryBlocks;
	}

	@Override
	public String toString() {
		if (matrixMetadata != null) {
			return matrixMetadata.toString();
		} else {
			return super.toString();
		}
	}
}
