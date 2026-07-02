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

/**
 * Enumeration of supported compression techniques for federated learning.
 * Used for configuration, serialization, and technique selection.
 */
public enum CompressionType {

	/** TopK sparsification: keep largest-magnitude elements only */
	TOPK("topk", "Top-K Sparsification"),

	/** Probabilistic quantization: reduce precision with stochastic rounding */
	PROBABILISTIC_QUANTIZATION("prob_quant", "Probabilistic Quantization"),

	/** No compression (passthrough) */
	NONE("none", "No Compression");

	private final String id;
	private final String description;

	CompressionType(String id, String description) {
		this.id = id;
		this.description = description;
	}

	public String getId() { return id; }
	public String getDescription() { return description; }

	/** Parse from string identifier (case-insensitive) */
	public static CompressionType fromString(String text) {
		for (CompressionType type : CompressionType.values()) {
			if (type.id.equalsIgnoreCase(text)) {
				return type;
			}
		}
		throw new IllegalArgumentException("Unknown compression type: " + text);
	}
}
