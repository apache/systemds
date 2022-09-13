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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ComputationCostEstimator extends ACostEstimate {

	private static final long serialVersionUID = -1205636215389161815L;

	/** The threshold before the commonValueImpact is starting. */
	private static final double cvThreshold = 0.2;

	private final InstructionTypeCounter ins;

	protected ComputationCostEstimator(InstructionTypeCounter counts) {
		this.ins = counts;

		if(LOG.isDebugEnabled())
			LOG.debug(this);
	}

	public ComputationCostEstimator(int scans, int decompressions, int overlappingDecompressions,
		int leftMultiplications, int rightMultiplications, int compressedMultiplication, int dictOps, int indexing,
		boolean isDensifying) {
		ins = new InstructionTypeCounter(scans, decompressions, overlappingDecompressions, leftMultiplications,
			rightMultiplications, compressedMultiplication, dictOps, indexing, isDensifying);

	}

	@Override
	protected double getCostSafe(CompressedSizeInfoColGroup g) {
		final int nVals = g.getNumVals();
		final int nCols = g.getColumns().length;
		final int nRows = g.getNumRows();
		// assume that it is never fully sparse
		final double sparsity = (nCols < 3 || ins.isDensifying()) ? 1 : g.getTupleSparsity() + 1E-10;

		final double commonFraction = g.getLargestOffInstances();

		if(g.isEmpty() && !ins.isDensifying())
			// set some small cost to empty
			return getCost(nRows, 1, nCols, 1, 0.00001);
		else if(g.isEmpty() || g.isConst())
			// const or densifying
			return getCost(nRows, 1, nCols, 1, 1);
		if(commonFraction > cvThreshold)
			return getCost(nRows, nRows - g.getLargestOffInstances(), nCols, nVals, sparsity);
		else
			return getCost(nRows, nRows, nCols, nVals, sparsity);
	}

	/**
	 * Get the cost of a column group.
	 * 
	 * @param nRows        The total number of rows
	 * @param nRowsScanned Number of rows to process in this specific group
	 * @param nCols        Number of columns in the column group
	 * @param nVals        Number of unique tuples contained in the group
	 * @param sparsity     The sparsity of the unique tuples stored
	 * @return A cost
	 */
	public double getCost(int nRows, int nRowsScanned, int nCols, int nVals, double sparsity) {
		sparsity = (nCols < 3 || sparsity > 0.4) ? 1 : sparsity;

		double cost = 0;
		cost += leftMultCost(nRowsScanned, nRows, nCols, nVals, sparsity);
		cost += scanCost(nRowsScanned, nCols, nVals, sparsity);
		cost += dictionaryOpsCost(nVals, nCols, sparsity);
		cost += rightMultCost(nCols, nVals, sparsity);
		cost += decompressionCost(nVals, nCols, nRowsScanned, sparsity);
		cost += overlappingDecompressionCost(nRowsScanned);
		cost += compressedMultiplicationCost(nRowsScanned, nRows, nVals, nCols, sparsity);
		cost += 100; // base cost
		if(cost < 0)
			throw new DMLCompressionException("Ivalid negative cost: " + cost);
		return cost;
	}

	public boolean isDense() {
		return ins.isDensifying();
	}

	@Override
	public double getCost(MatrixBlock mb) {
		double cost = 0;
		final double nCols = mb.getNumColumns();
		final double nRows = mb.getNumRows();
		final double sparsity = (nCols < 3 || ins.isDensifying()) ? 1 : mb.getSparsity();

		cost += dictionaryOpsCost(nRows, nCols, sparsity);
		// Naive number of floating point multiplications
		cost += leftMultCost(0, nRows * nCols * sparsity + nCols);
		cost += rightMultCost(nRows * nCols * sparsity, nRows * nCols);
		// Scan cost we set the rows scanned to zero, since they
		// are not indirectly scanned like in compression
		cost += scanCost(0, nRows, nCols, sparsity);
		cost += compressedMultiplicationCost(0, 0, nRows, nCols, sparsity);
		// decompression cost ... 0 for both overlapping and normal decompression

		if(cost < 0)
			throw new DMLCompressionException("Invalid negative cost : " + cost);
		return cost;
	}

	@Override
	public double getCost(AColGroup cg, int nRows) {
		return cg.getCost(this, nRows);
	}

	@Override
	public boolean shouldSparsify() {
		return ins.getLeftMultiplications() > 0 || ins.getCompressedMultiplications() > 0 ||
			ins.getRightMultiplications() > 0;
	}

	private double dictionaryOpsCost(double nVals, double nCols, double sparsity) {
		// Dictionary ops simply goes through dictionary and modify all values.
		// Therefore the cost is in number of cells in the dictionary.
		// * 2 because we allocate a output of same size at least
		return ins.getDictionaryOps() * sparsity * nVals * nCols * 2;
	}

	private double leftMultCost(double nRowsScanned, double nRows, double nCols, double nVals, double sparsity) {
		// Plus nVals * 2 because of allocation of nVals array and scan of that
		final double preScalingCost = Math.max(nRowsScanned, nRows / 10) + nVals * 2;
		final double postScalingCost = sparsity * nVals * nCols;
		return leftMultCost(preScalingCost, postScalingCost);
	}

	private double leftMultCost(double preAggregateCost, double postScalingCost) {
		return ins.getLeftMultiplications() * (preAggregateCost + postScalingCost);
	}

	private double rightMultCost(double nVals, double nCols, double sparsity) {
		final double preMultiplicationCost = sparsity * nCols * nVals;
		final double allocationCost = nVals;
		return rightMultCost(preMultiplicationCost, allocationCost);
	}

	private double rightMultCost(double preMultiplicationCost, double allocationCost) {
		return ins.getRightMultiplications() * (preMultiplicationCost + allocationCost);
	}

	private double decompressionCost(double nVals, double nCols, double nRowsScanned, double sparsity) {
		return ins.getDecompressions() * (nCols * nRowsScanned * sparsity);
	}

	private double overlappingDecompressionCost(double nRows) {
		return ins.getOverlappingDecompressions() * nRows;
	}

	private double scanCost(double nRowsScanned, double nVals, double nCols, double sparsity) {
		return ins.getScans() * (nRowsScanned + nVals * nCols * sparsity);
	}

	private double compressedMultiplicationCost(double nRowsScanned, double nRows, double nVals, double nCols,
		double sparsity) {
		// return _compressedMultiplication * Math.max(nRowsScanned * nCols ,nVals * nCols * sparsity );
		return ins.getCompressedMultiplications() * (Math.max(nRowsScanned, nRows / 10) + nVals * nCols * sparsity);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(" : ");
		sb.append(ins.toString());
		return sb.toString();
	}
}
