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

package org.apache.sysds.runtime.controlprogram.federated.compression.Quantization;

import org.apache.sysds.runtime.controlprogram.federated.compression.CompressedMatrix;
import org.apache.sysds.runtime.controlprogram.federated.compression.CompressionType;
import org.apache.sysds.runtime.controlprogram.federated.compression.MatrixCompressor;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.CompressionException;
import org.apache.sysds.runtime.controlprogram.federated.compression.exceptions.DecompressionException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.Random;

/**
 * Probabilistic Quantization Compressor.
 *
 * Reduces numerical precision using stochastic rounding to maintain an unbiased estimator — meaning E[quantized] =
 * original on average. This is critical for federated learning convergence guarantees.
 *
 * Supports 2, 4, or 8 bits per value: 2-bit → 4 levels → 16x compression vs 32-bit float 4-bit → 16 levels → 8x
 * compression 8-bit → 256 levels → 4x compression
 */
public class ProbabilisticQuantizationCompressor implements MatrixCompressor {

	private final int bitsPerValue; // 2, 4, or 8
	private final Random rng;

	public ProbabilisticQuantizationCompressor(int bitsPerValue) {
		if(bitsPerValue != 2 && bitsPerValue != 4 && bitsPerValue != 8) {
			throw new IllegalArgumentException("bitsPerValue must be 2, 4, or 8");
		}
		this.bitsPerValue = bitsPerValue;
		this.rng = new Random(42); // Fixed seed for reproducibility
	}

	@Override
	public CompressedMatrix compress(MatrixBlock input) throws CompressionException {
		try {
			int numRows = input.getNumRows();
			int numCols = input.getNumColumns();
			int totalElements = numRows * numCols;

			// Find min and max for normalization
			double[] minMax = findMinMax(input, numRows, numCols);
			double min = minMax[0];
			double max = minMax[1];

			int levels = 1 << bitsPerValue; // 2^bits

			// Quantize each element probabilistically
			byte[] quantized = new byte[totalElements];
			for(int i = 0; i < numRows; i++) {
				for(int j = 0; j < numCols; j++) {
					double value = input.get(i, j);
					quantized[i * numCols + j] = probabilisticRound(value, min, max, levels);
				}
			}

			double ratio = 32.0 / bitsPerValue; // vs 32-bit float

			QuantizedData data = new QuantizedData(quantized, min, max, levels, bitsPerValue, numRows, numCols);

			return new CompressedMatrix(CompressionType.PROBABILISTIC_QUANTIZATION, numRows, numCols, data, ratio);

		}
		catch(Exception e) {
			throw new CompressionException("Quantization compression failed: " + e.getMessage(), e);
		}
	}

	@Override
	public MatrixBlock decompress(CompressedMatrix compressed) throws DecompressionException {
		try {
			QuantizedData data = (QuantizedData) compressed.getCompressedData();
			MatrixBlock result = new MatrixBlock(data.numRows, data.numCols, false);
			result.allocateDenseBlock();

			for(int i = 0; i < data.numRows; i++) {
				for(int j = 0; j < data.numCols; j++) {
					byte levelIndex = data.quantizedValues[i * data.numCols + j];
					double value = data.reconstructValue(levelIndex);
					result.set(i, j, value);
				}
			}

			result.examSparsity();
			return result;

		}
		catch(ClassCastException e) {
			throw new DecompressionException("Invalid compressed data type for Quantization", e);
		}
		catch(Exception e) {
			throw new DecompressionException("Quantization decompression failed: " + e.getMessage(), e);
		}
	}

	// -----------------------------------------------------------------------
	// Private helpers
	// -----------------------------------------------------------------------

	/**
	 * Stochastic rounding: for value x between levels q_i and q_{i+1}: P(round up) = (x - q_i) / (q_{i+1} - q_i)
	 * P(round down) = 1 - P(round up) This gives E[output] = x (unbiased).
	 */
	private byte probabilisticRound(double value, double min, double max, int levels) {
		// Handle constant matrix edge case
		if(max - min < 1e-10)
			return 0;

		// Normalize to [0, 1]
		double normalized = (value - min) / (max - min);

		// Find bounding level indices
		double scaled = normalized * (levels - 1);
		int lowerIdx = (int) scaled;
		int upperIdx = lowerIdx + 1;

		if(lowerIdx == upperIdx) {
			return (byte) lowerIdx;
		}

		// Probabilistic decision
		double probUp = scaled - lowerIdx;
		return (rng.nextDouble() < probUp) ? (byte) upperIdx : (byte) lowerIdx;
	}

	/** Find min and max values across the entire matrix */
	private double[] findMinMax(MatrixBlock input, int numRows, int numCols) {
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;

		for(int i = 0; i < numRows; i++) {
			for(int j = 0; j < numCols; j++) {
				double val = input.get(i, j);
				if(val < min)
					min = val;
				if(val > max)
					max = val;
			}
		}

		// Handle all-zero matrix
		if(min == Double.MAX_VALUE) {
			min = 0;
			max = 0;
		}
		return new double[] {min, max};
	}
}
