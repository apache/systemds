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
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Exact compressed size estimator (examines entire dataset).
 */
public class CompressedSizeEstimatorExact extends CompressedSizeEstimator {

	public CompressedSizeEstimatorExact(MatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int estimate,
		int nrUniqueUpperBound) {
		final int _numRows = getNumRows();
		final IEncode map = IEncode.createFromMatrixBlock(_data, _cs.transposed, colIndexes);
		final EstimationFactors em = map.computeSizeEstimation(colIndexes, _numRows, _data.getSparsity(),
			_data.getSparsity());
		return new CompressedSizeInfoColGroup(colIndexes, em, _cs.validCompressions, map);
	}

	@Override
	protected CompressedSizeInfoColGroup estimateJoinCompressedSize(int[] joined, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int joinedMaxDistinct) {
		final int _numRows = getNumRows();
		final IEncode map = g1.getMap().join(g2.getMap());
		EstimationFactors em = null;
		if(map != null)
			em = map.computeSizeEstimation(joined, _numRows, _data.getSparsity(), _data.getSparsity());
		if(em == null)
			em = EstimationFactors.emptyFactors(joined.length, _numRows);

		return new CompressedSizeInfoColGroup(joined, em, _cs.validCompressions, map);
	}

	@Override
	protected int worstCaseUpperBound(int[] columns) {
		return getNumRows();
	}

	@Override
	public final int getSampleSize() {
		return getNumRows();
	}
}
