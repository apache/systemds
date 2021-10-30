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

public class HybridCostEstimator implements ICostEstimate {

	private static final long serialVersionUID = -542307595058927576L;

	final ComputationCostEstimator costEstimator;
	final MemoryCostEstimator memoryCostEstimator;

	protected HybridCostEstimator(int nRows, int nCols, double sparsity, InstructionTypeCounter counts) {
		costEstimator = new ComputationCostEstimator(nRows, nCols, sparsity, counts);
		memoryCostEstimator = new MemoryCostEstimator(nRows, nCols, sparsity);
	}

	protected HybridCostEstimator(int nRows, int nCols, double sparsity, int scans, int decompressions,
		int overlappingDecompressions, int leftMultiplictions, int compressedMultiplication, int rightMultiplications,
		int dictioanaryOps, boolean isDensifying) {
		costEstimator = new ComputationCostEstimator(nRows, nCols, sparsity, scans, decompressions,
			overlappingDecompressions, leftMultiplictions, compressedMultiplication, rightMultiplications,
			dictioanaryOps, isDensifying);
		memoryCostEstimator = new MemoryCostEstimator(nRows, nCols, sparsity);
	}

	@Override
	public double getUncompressedCost(CompressedSizeInfoColGroup g) {
		double cost = costEstimator.getUncompressedCost(g);
		// not multiplying with uncompressed, since that would be multiplying with 1
		return cost;
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		double cost = costEstimator.getCostOfColumnGroup(g);
		// multiplying with compression ratio since this scale cost with size.
		cost *= calculateCompressionRatio(g);
		return cost;
	}

	private double calculateCompressionRatio(CompressedSizeInfoColGroup g) {
		double denseSize = memoryCostEstimator.getUncompressedCost(g);
		double compressedSize = memoryCostEstimator.getCostOfColumnGroup(g);
		// If the compression increase size then the fraction is above 1, aka the cost should be significantly smaller.
		return compressedSize / denseSize;
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}
}
