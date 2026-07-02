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

import org.apache.sysds.runtime.controlprogram.federated.compression.TopK.TopKCompressor;
import org.apache.sysds.runtime.controlprogram.federated.compression.Quantization.ProbabilisticQuantizationCompressor;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.CompressionException;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.DecompressionException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Factory for creating compressor instances from configuration.
 * Centralizes compressor instantiation and parameter validation.
 *
 * Usage:
 *   CompressionConfig config = CompressionConfig.builder()
 *       .enable(true)
 *       .withType(CompressionType.TOPK)
 *       .withSparsity(0.01)
 *       .build();
 *   MatrixCompressor compressor = CompressionFactory.create(config);
 */
public class CompressionFactory {

	private CompressionFactory() {
		// Utility class — no instantiation
	}

	/**
	 * Create a compressor from a CompressionConfig.
	 * @param config The compression configuration
	 * @return A ready-to-use MatrixCompressor
	 * @throws IllegalArgumentException if the config is invalid
	 */
	public static MatrixCompressor create(CompressionConfig config) {
		if(config == null || !config.isEnabled())
			return new PassthroughCompressor();
		switch(config.getType()) {
			case TOPK:
				return new TopKCompressor(config.getSparsity(), true);
			case PROBABILISTIC_QUANTIZATION:
				return new ProbabilisticQuantizationCompressor(config.getBits());
			case NONE:
			default:
				return new PassthroughCompressor();
		}
	}

	// -----------------------------------------------------------------------
	// Passthrough compressor (no-op) for when compression is disabled
	// -----------------------------------------------------------------------

	/**
	 * No-op compressor: returns the matrix as-is.
	 * Used when compression is disabled or type is NONE.
	 */
	private static class PassthroughCompressor implements MatrixCompressor {

		@Override
		public CompressedMatrix compress(MatrixBlock input)
				throws CompressionException {
			return new CompressedMatrix(
				CompressionType.NONE,
				input.getNumRows(),
				input.getNumColumns(),
				input,
				1.0
			);
		}

		@Override
		public MatrixBlock decompress(CompressedMatrix compressed)
				throws DecompressionException {
			return (MatrixBlock) compressed.getCompressedData();
		}
	}
}
