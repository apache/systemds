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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public abstract class AColumnCoCoder {

	protected static final Log LOG = LogFactory.getLog(AColumnCoCoder.class.getName());

	final protected CompressedSizeEstimator _sest;
	final protected ICostEstimate _cest;
	final protected CompressionSettings _cs;

	protected AColumnCoCoder(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator,
		CompressionSettings cs) {
		_sest = sizeEstimator;
		_cest = costEstimator;
		_cs = cs;
	}

	/**
	 * CoCode columns into groups.
	 * 
	 * @param colInfos The current individually sampled and evaluated column groups.
	 * @param k        The parallelization available to the underlying implementation.
	 * @return CoCoded column groups.
	 */
	protected abstract CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k);

	protected CompressedSizeInfoColGroup join(int[] joined, CompressedSizeInfoColGroup lhs,
		CompressedSizeInfoColGroup rhs, boolean analyze) {
		return analyze ? _sest.estimateJoinCompressedSize(joined, lhs, rhs) : joinWithoutAnalysis(joined, lhs, rhs);
	}

	protected CompressedSizeInfoColGroup joinWithoutAnalysis(int[] joined, CompressedSizeInfoColGroup lhs,
		CompressedSizeInfoColGroup rhs) {
		final int lhsV = lhs.getNumVals();
		final int rhsV = rhs.getNumVals();
		final int joinedMaxDistinct = (int) Math.min((long) lhsV * (long) rhsV, (long) _sest.getNumRows());
		return new CompressedSizeInfoColGroup(joined, joinedMaxDistinct, _sest.getNumRows());
	}

	protected CompressedSizeInfoColGroup analyze(CompressedSizeInfoColGroup g) {
		if(g.getBestCompressionType() == null)
			return _sest.estimateCompressedColGroupSize(g.getColumns());
		else
			return g;
	}
}
