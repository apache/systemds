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

package org.apache.sysds.runtime.controlprogram.federated.compression.TopK;

import java.io.Serializable;

/**
 * Immutable container for TopK-compressed matrix data.
 * Stores only the K largest-magnitude elements with their positions,
 * designed for efficient serialization across federated workers.
 */
public class TopKData implements Serializable {

	private static final long serialVersionUID = 1L;

	public final int[] indices;    // Linear indices of kept elements (row*numCols + col)
	public final double[] values;  // Corresponding original values
	public final int numCols;      // Needed for index → (row, col) conversion

	public TopKData(int[] indices, double[] values, int numCols) {
		if (indices.length != values.length) {
			throw new IllegalArgumentException(
				"Indices and values arrays must have the same length");
		}
		this.indices = indices.clone();  // Defensive copy
		this.values = values.clone();
		this.numCols = numCols;
	}

	/** Number of kept elements */
	public int size() {
		return indices.length;
	}

	/** Estimate serialized size in bytes (4 bytes per int + 8 bytes per double) */
	public long estimateSizeBytes() {
		return (long) indices.length * 12 + 64;  // +64 for object headers
	}

	@Override
	public String toString() {
		return String.format("TopKData[k=%d, numCols=%d]", indices.length, numCols);
	}
}
