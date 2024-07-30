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

package org.apache.sysds.runtime.frame.data.compress;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;

public class ArrayCompressionStatistics {

	public final long originalSize;
	public final long compressedSizeEstimate;
	public final boolean shouldCompress;
	public final ValueType valueType;
	public final boolean containsNull;
	public final boolean sampledAllRows;
	public final FrameArrayType bestType;
	public final int bytePerValue;
	public final int nUnique;

	public ArrayCompressionStatistics(int bytePerValue, int nUnique, boolean shouldCompress, ValueType valueType,
		boolean containsNull, FrameArrayType bestType, long originalSize, long compressedSizeEstimate,
		boolean sampledAllRows) {
		this.bytePerValue = bytePerValue;
		this.nUnique = nUnique;
		this.shouldCompress = shouldCompress;
		this.valueType = valueType;
		this.containsNull = containsNull;
		this.bestType = bestType;
		this.originalSize = originalSize;
		this.compressedSizeEstimate = compressedSizeEstimate;
		this.sampledAllRows = sampledAllRows;

		if(valueType == null)
			throw new RuntimeException("Invalid null valuetype in statistics");
	}

	@Override
	public String toString() {
		if(!sampledAllRows)
			return String.format(
				"Compressed Stats: size:%8d->%8d, Use:%10s, EstUnique:%6d, ValueType:%7s, ContainsNull:%5s", originalSize,
				compressedSizeEstimate, bestType(), nUnique, valueType, containsNull);
		else
			return String.format("Compressed Stats: size:%8d->%8d, Use:%10s, Unique:%6d, ValueType:%7s, ContainsNull:%5s",
				originalSize, compressedSizeEstimate, bestType(), nUnique, valueType, containsNull);
	}

	private String bestType() {
		return bestType == null ? "None" : bestType.toString();
	}
}
