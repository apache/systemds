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

package org.apache.sysds.runtime.compress.estim;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.lib.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Exact compressed size estimator (examines entire dataset).
 */
public class CompressedSizeEstimatorExact extends CompressedSizeEstimator {

	public CompressedSizeEstimatorExact(MatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int nrUniqueUpperBound) {
		// exact estimator can ignore upper bound.
		ABitmap entireBitMap = BitmapEncoder.extractBitmap(colIndexes, _data, _transposed);
		return new CompressedSizeInfoColGroup(estimateCompressedColGroupSize(entireBitMap, colIndexes),
			_cs.validCompressions);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" transposed: ");
		sb.append(_transposed);
		sb.append(" cols: ");
		sb.append(_numCols);
		sb.append(" rows: ");
		sb.append(_numRows);
		return sb.toString();
	}
}
