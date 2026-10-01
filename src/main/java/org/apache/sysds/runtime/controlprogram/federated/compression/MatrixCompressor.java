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

package org.apache.sysds.runtime.controlprogram.federated.compression;

import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.CompressionException;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.DecompressionException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Interface for matrix compression techniques in federated learning. All compressors must implement compress/decompress
 * operations.
 */
public interface MatrixCompressor {

	/**
	 * Compress a matrix block for transmission.
	 *
	 * @param input The source matrix to compress
	 * @return CompressedMatrix containing compressed data and metadata
	 * @throws CompressionException if compression fails
	 */
	CompressedMatrix compress(MatrixBlock input) throws CompressionException;

	/**
	 * Decompress a compressed matrix back to MatrixBlock.
	 *
	 * @param compressed The compressed data to decompress
	 * @return Reconstructed MatrixBlock (may be approximate)
	 * @throws DecompressionException if decompression fails
	 */
	MatrixBlock decompress(CompressedMatrix compressed) throws DecompressionException;

	/**
	 * Estimate the compression ratio achieved. Higher is better (e.g. 10.0 means 10x smaller).
	 */
	default double estimateCompressionRatio(long originalSize, long compressedSize) {
		return compressedSize == 0 ? Double.MAX_VALUE : (double) originalSize / compressedSize;
	}
}
