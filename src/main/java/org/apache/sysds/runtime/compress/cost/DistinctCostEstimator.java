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

import java.util.Collection;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

/**
 * A Cost based estimator that based the compression and co-coding cost on the number of distinct elements in the column
 * groups.
 * 
 * The base cost of the uncompressed representation is the number of cells in the matrix that has a value. Aka nonzero
 * values.
 */
public class DistinctCostEstimator implements ICostEstimate {

	private final static int toSmallForAnalysis = 64;
	private final int nRows;

	/**
	 * This value specifies the maximum distinct count allowed int a coCoded group. Note that this value is the number
	 * of distinct tuples not the total number of values. That value can be calculated by multiplying the number of
	 * tuples with columns in the coCoded group.
	 */
	private final double largestDistinct;

	public DistinctCostEstimator(int nRows, CompressionSettings cs) {
		this.largestDistinct = Math.min(4096, Math.max(256, (int) (nRows * cs.coCodePercentage)));
		this.nRows = nRows;
	}

	@Override
	public double getUncompressedCost(int nRows, int nCols, int sparsity) {
		return nRows * nCols * sparsity;
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		int nVals = g.getNumVals();
		return nVals < largestDistinct ? nVals : Double.POSITIVE_INFINITY;
	}

	@Override
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gs) {
		long distinct = 0;
		for(CompressedSizeInfoColGroup g : gs)
			distinct += g.getNumVals();

		return distinct;
	}

	@Override
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gs, CompressedSizeInfoColGroup g) {
		long distinct = 0;
		for(CompressedSizeInfoColGroup ge : gs)
			distinct += ge.getNumVals();
		distinct += g.getNumVals();
		return distinct;
	}

	@Override
	public double getCostOfTwoGroups(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		throw new DMLCompressionException("Should not be called");
	}

	@Override
	public boolean isCompareAll() {
		return false;
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return g1.getNumVals() * g2.getNumVals() < toSmallForAnalysis;
	}

	@Override
	public boolean shouldTryJoin(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return g1.getNumVals() * g2.getNumVals() < nRows;
	}
}
