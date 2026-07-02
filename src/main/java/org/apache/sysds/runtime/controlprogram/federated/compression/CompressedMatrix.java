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

import java.io.Serializable;

/**
 * Generic container for compressed matrix data.
 * Stores the compressed representation along with metadata
 * needed for decompression and size estimation.
 */
public class CompressedMatrix implements Serializable {

	private static final long serialVersionUID = 1L;

	private final CompressionType type;
	private final int numRows;
	private final int numCols;
	private final Object compressedData;   // Technique-specific data
	private final double compressionRatio;
	private final byte[] metadata;         // Optional: scaling factors, etc.

	public CompressedMatrix(CompressionType type, int numRows, int numCols,
					        Object compressedData, double compressionRatio) {
		this(type, numRows, numCols, compressedData, compressionRatio, null);
	}

	public CompressedMatrix(CompressionType type, int numRows, int numCols,
					        Object compressedData, double compressionRatio,
					        byte[] metadata) {
		this.type = type;
		this.numRows = numRows;
		this.numCols = numCols;
		this.compressedData = compressedData;
		this.compressionRatio = compressionRatio;
		this.metadata = metadata;
	}

	public CompressionType getType() { return type; }
	public int getNumRows() { return numRows; }
	public int getNumCols() { return numCols; }
	public Object getCompressedData() { return compressedData; }
	public double getCompressionRatio() { return compressionRatio; }
	public byte[] getMetadata() { return metadata; }

	/** Estimate original size in bytes (8 bytes per double) */
	public long estimateOriginalSizeBytes() {
		return (long) numRows * numCols * 8;
	}

	/** Estimate compressed size in bytes */
	public long getCompressedSizeBytes() {
		if (compressedData instanceof byte[]) {
			return ((byte[]) compressedData).length;
		}
		return 0;
	}

	@Override
	public String toString() {
		return String.format("CompressedMatrix[%s, %dx%d, ratio=%.2fx]",
			type.getId(), numRows, numCols, compressionRatio);
	}
}
