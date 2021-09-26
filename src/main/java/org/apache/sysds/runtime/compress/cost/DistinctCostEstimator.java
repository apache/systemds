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
	private static final long serialVersionUID = 4784682182584508597L;
	private final static int toSmallForAnalysis = 64;

	/**
	 * This value specifies the maximum distinct count allowed int a coCoded group. Note that this value is the number
	 * of distinct tuples not the total number of values. That value can be calculated by multiplying the number of
	 * tuples with columns in the coCoded group.
	 */
	private final double largestDistinct;

	private final int nRows;
	private final double sparsity;

	public DistinctCostEstimator(int nRows, CompressionSettings cs, double sparsity) {
		this.largestDistinct = Math.min(4096, Math.max(256, (int) (nRows * cs.coCodePercentage)));
		this.nRows = nRows;
		this.sparsity = sparsity;
	}

	@Override
	public double getUncompressedCost(CompressedSizeInfoColGroup g) {
		final int nCols = g.getColumns().length;
		return nRows * nCols * sparsity;
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		if(g == null)
			return Double.POSITIVE_INFINITY;
		int nVals = Math.max(g.getNumVals(), toSmallForAnalysis);
		return nVals < largestDistinct ? nVals : Double.POSITIVE_INFINITY;
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return g1.getNumVals() * g2.getNumVals() < toSmallForAnalysis;
	}
}
