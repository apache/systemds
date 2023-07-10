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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * A Cost based estimator that based the compression and co-coding cost on the number of distinct elements in the column
 * groups.
 * 
 * The base cost of the uncompressed representation is the number of cells in the matrix that has a value. Aka nonzero
 * values.
 */
public class DistinctCostEstimator extends ACostEstimate {
	private static final long serialVersionUID = 4784682182584508597L;
	private final static int toSmallForAnalysis = 64;
	private final double largestDistinct;

	public DistinctCostEstimator(int nRows, CompressionSettings cs, double sparsity) {
		this.largestDistinct = Math.min(4096, Math.max(256, (int) (nRows * cs.coCodePercentage)));
	}

	@Override
	protected double getCostSafe(CompressedSizeInfoColGroup g) {
		int nVals = Math.max(g.getNumVals(), toSmallForAnalysis);
		return nVals < largestDistinct ? nVals : Double.POSITIVE_INFINITY;
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
