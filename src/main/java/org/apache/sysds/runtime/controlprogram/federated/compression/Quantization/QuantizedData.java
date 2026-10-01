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

import java.io.Serializable;

/**
 * Immutable container for probabilistically quantized matrix data. Stores quantized byte indices and the scaling
 * parameters needed to reconstruct approximate original values on decompression.
 */
public class QuantizedData implements Serializable {

	private static final long serialVersionUID = 1L;

	public final byte[] quantizedValues; // Quantized level indices
	public final double min; // Original minimum value
	public final double max; // Original maximum value
	public final int levels; // Number of quantization levels (2^bits)
	public final int bitsPerValue; // Bits used per element
	public final int numRows;
	public final int numCols;

	public QuantizedData(byte[] quantizedValues, double min, double max, int levels, int bitsPerValue, int numRows,
		int numCols) {
		this.quantizedValues = quantizedValues.clone(); // Defensive copy
		this.min = min;
		this.max = max;
		this.levels = levels;
		this.bitsPerValue = bitsPerValue;
		this.numRows = numRows;
		this.numCols = numCols;
	}

	/** Number of quantized elements */
	public int size() {
		return quantizedValues.length;
	}

	/** Estimate serialized size in bytes */
	public long estimateSizeBytes() {
		return quantizedValues.length + 64; // +64 for scalar fields and headers
	}

	/** Reconstruct a double value from a quantized level index */
	public double reconstructValue(byte levelIndex) {
		if(max - min < 1e-10)
			return min; // Constant matrix
		int idx = levelIndex & 0xFF; // Treat byte as unsigned
		return min + (idx / (double) (levels - 1)) * (max - min);
	}

	@Override
	public String toString() {
		return String.format("QuantizedData[%dx%d, levels=%d, bits=%d, min=%.4f, max=%.4f]", numRows, numCols, levels,
			bitsPerValue, min, max);
	}
}
