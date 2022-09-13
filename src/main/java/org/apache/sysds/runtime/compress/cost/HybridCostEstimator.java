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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class HybridCostEstimator extends ACostEstimate {

	private static final long serialVersionUID = -542307595058927576L;

	final ComputationCostEstimator costEstimator;
	final MemoryCostEstimator memoryCostEstimator;

	protected HybridCostEstimator(InstructionTypeCounter counts) {
		costEstimator = new ComputationCostEstimator(counts);
		memoryCostEstimator = new MemoryCostEstimator();
	}

	protected HybridCostEstimator(int scans, int decompressions, int overlappingDecompressions, int leftMultiplications,
		int compressedMultiplication, int rightMultiplications, int dictionaryOps, int indexing, boolean isDensifying) {
		costEstimator = new ComputationCostEstimator(scans, decompressions, overlappingDecompressions,
			leftMultiplications, compressedMultiplication, rightMultiplications, dictionaryOps, indexing, isDensifying);
		memoryCostEstimator = new MemoryCostEstimator();
	}

	@Override
	protected double getCostSafe(CompressedSizeInfoColGroup g) {
		final double cost = costEstimator.getCostSafe(g);
		final double denseSize = g.getNumRows() * g.getColumns().length * 8;
		final double compressedSize = memoryCostEstimator.getCostSafe(g);
		return cost * (compressedSize / denseSize);
	}

	@Override
	public double getCost(MatrixBlock mb) {
		throw new NotImplementedException();
	}

	@Override
	public double getCost(AColGroup cg, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public boolean shouldSparsify() {
		return false;
	}
}
