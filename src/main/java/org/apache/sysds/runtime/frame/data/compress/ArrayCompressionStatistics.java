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
	public final FrameArrayType bestType;
	public final int bitPerValue;
	public final int nUnique;

	public ArrayCompressionStatistics(int bitPerValue, int nUnique, boolean shouldCompress, ValueType valueType,
		FrameArrayType bestType, long originalSize, long compressedSizeEstimate) {
		this.bitPerValue = bitPerValue;
		this.nUnique = nUnique;
		this.shouldCompress = shouldCompress;
		this.valueType = valueType;
		this.bestType = bestType;
		this.originalSize = originalSize;
		this.compressedSizeEstimate = compressedSizeEstimate;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("Compressed Stats: size:%8d->%8d, Use:%10s, Unique:%6d, ValueType:%7s", originalSize,
			compressedSizeEstimate, bestType.toString(), nUnique, valueType));
		return sb.toString();
	}
}
