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

package org.apache.sysds.runtime.compress.cocode;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public abstract class AColumnGroupPartitioner {

	protected CompressedSizeEstimator _est;
	protected CompressionSettings _cs;
	protected int _numRows;

	protected AColumnGroupPartitioner(CompressedSizeEstimator sizeEstimator, CompressionSettings cs, int numRows) {
		_est = sizeEstimator;
		_cs = cs;
		_numRows = numRows;
	}

	public abstract CompressedSizeInfo partitionColumns(CompressedSizeInfo colInfos);

	protected CompressedSizeInfoColGroup join(CompressedSizeInfoColGroup lhs, CompressedSizeInfoColGroup rhs) {
		// TODO optimzie so that we do not allocate all these small int arrays.
		// Note this is usually 1 - 2% of the compression time, so this optimization would only benefit
		// the compression time if there is a large number of columns, and even then it would most likely
		// only benefit the amount of memory used slightly.
		int[] lhsCols = lhs.getColumns();
		int[] rhsCols = rhs.getColumns();
		int[] joined = new int[lhsCols.length + rhsCols.length];
		System.arraycopy(lhsCols, 0, joined, 0, lhsCols.length);
		System.arraycopy(rhsCols, 0, joined, lhsCols.length, rhsCols.length);
		Arrays.sort(joined);
		return _est.estimateCompressedColGroupSize(joined);
	}
}
