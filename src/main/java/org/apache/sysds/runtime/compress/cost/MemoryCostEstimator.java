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

package org.apache.sysds.runtime.compress.cost;

import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class MemoryCostEstimator implements ICostEstimate {
	private static final long serialVersionUID = -1264988969161809465L;

	private final int nRows;
	// private final int nCols;
	private final double sparsity;

	public MemoryCostEstimator(int nRows, int nCols, double sparsity) {
		this.nRows = nRows;
		// this.nCols = nCols;
		this.sparsity = sparsity;
	}

	@Override
	public double getUncompressedCost(CompressedSizeInfoColGroup g) {
		return MatrixBlock.estimateSizeInMemory(nRows, g.getColumns().length, sparsity);
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		if(g == null)
			return Double.POSITIVE_INFINITY;
		return g.getMinSize();
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}
}
