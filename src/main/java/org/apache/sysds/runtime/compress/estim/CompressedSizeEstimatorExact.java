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
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Exact compressed size estimator (examines entire dataset).
 */
public class CompressedSizeEstimatorExact extends CompressedSizeEstimator {

	protected CompressedSizeEstimatorExact(MatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int estimate,
		int nrUniqueUpperBound) {
		// exact estimator can ignore upper bound since it returns the accurate values.
		final ABitmap entireBitMap = BitmapEncoder.extractBitmap(colIndexes, _data, _cs.transposed, estimate, false);
		EstimationFactors em = null;
		if(entireBitMap != null)
			em = estimateCompressedColGroupSize(entireBitMap, colIndexes);
		if(em == null)
			em = EstimationFactors.emptyFactors(colIndexes.length, getNumRows());

		return new CompressedSizeInfoColGroup(colIndexes, em, _cs.validCompressions, entireBitMap, getNumRows());
	}

	@Override
	protected CompressedSizeInfoColGroup estimateJoinCompressedSize(int[] joined, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int joinedMaxDistinct) {
		final int _numRows = getNumRows();
		AMapToData map = MapToFactory.join(g1.getMap(), g2.getMap());
		EstimationFactors em = null;
		if(map != null)
			em = EstimationFactors.computeSizeEstimation(joined, map, _cs.validCompressions.contains(CompressionType.RLE),
				_numRows, false);

		if(em == null)
			em = EstimationFactors.emptyFactors(joined.length, getNumRows());

		return new CompressedSizeInfoColGroup(joined, em, _cs.validCompressions, map);
	}

	@Override
	protected int worstCaseUpperBound(int[] columns) {
		return getNumRows();
	}
}
