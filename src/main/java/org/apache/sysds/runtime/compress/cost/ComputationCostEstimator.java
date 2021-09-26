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
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public class ComputationCostEstimator implements ICostEstimate {

	private static final long serialVersionUID = -1205636215389161815L;

	/** A factor of how much common values can be exploited. */
	private static final double commonValueImpact = 0.75;
	/** The threshold before the commonValueImpact is tarting. */
	private static final double cvThreshold = 0.2;

	/** A factor for when the number of distinct tuples start scaling the cost. */
	private static final int scalingStart = 1000;

	private final int _nRows;
	private final int _nCols;
	// private final double _sparsity;

	private final int _scans;
	private final int _decompressions;
	private final int _overlappingDecompressions;
	private final int _leftMultiplications;
	private final int _rightMultiplications;
	private final int _compressedMultiplication;
	private final int _dictionaryOps;

	/** Boolean specifying if the matrix is getting densified, meaning exploiting zeros is gone. */
	private final boolean _isDensifying;

	protected ComputationCostEstimator(int nRows, int nCols, double sparsity, InstructionTypeCounter counts) {
		_nRows = nRows;
		_nCols = nCols;
		// _sparsity = sparsity;
		_scans = counts.scans;
		_decompressions = counts.decompressions;
		_overlappingDecompressions = counts.overlappingDecompressions;
		_leftMultiplications = counts.leftMultiplications;
		_compressedMultiplication = counts.compressedMultiplications;
		_rightMultiplications = counts.rightMultiplications;
		_dictionaryOps = counts.dictionaryOps;
		_isDensifying = counts.isDensifying;
		// _rowBasedOps = counts.rowBasedOps;
		if(LOG.isDebugEnabled())
			LOG.debug(this);
	}

	public ComputationCostEstimator(int nRows, int nCols, double sparsity, int scans, int decompressions,
		int overlappingDecompressions, int leftMultiplictions, int compressedMultiplication, int rightMultiplications,
		int dictioanaryOps, boolean isDensifying) {
		_nRows = nRows;
		_nCols = nCols;
		// _sparsity = sparsity;
		_scans = scans;
		_decompressions = decompressions;
		_overlappingDecompressions = overlappingDecompressions;
		_leftMultiplications = leftMultiplictions;
		_compressedMultiplication = compressedMultiplication;
		_rightMultiplications = rightMultiplications;
		_dictionaryOps = dictioanaryOps;
		_isDensifying = isDensifying;
	}

	@Override
	public double getUncompressedCost(CompressedSizeInfoColGroup g) {
		throw new NotImplementedException();
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		if(g == null)
			return Double.POSITIVE_INFINITY;
		double cost = 0;
		cost += _scans * scanCost(g);
		cost += _decompressions * decompressionCost(g);
		cost += _overlappingDecompressions * overlappingDecompressionCost(g);

		final int rowsCols = 16;

		final double scalingFactor = getScalingFactor(g.getNumVals());
		cost += _leftMultiplications * leftMultCost(g) * rowsCols;
		cost += _rightMultiplications * rightMultCost(g) * rowsCols;
		cost += _dictionaryOps * dictionaryOpsCost(g);
		cost += _compressedMultiplication * _compressedMultCost(g) * rowsCols;

		return cost * scalingFactor;
	}

	private double getScalingFactor(double nrValues) {
		double scalingFactor = 1;
		if(nrValues > scalingStart)
			scalingFactor += ((nrValues - scalingStart) / scalingStart) * 0.01;
		if(nrValues > Character.MAX_VALUE)
			scalingFactor += .5; // double cost if the dictionaries have to contain chars.
		return scalingFactor;
	}

	private double scanCost(CompressedSizeInfoColGroup g) {
		return _nRows;
	}

	private double leftMultCost(CompressedSizeInfoColGroup g) {
		final int nColsInGroup = g.getColumns().length;
		final double mcf = g.getMostCommonFraction();
		final double preAggregateCost = (mcf > cvThreshold ? _nRows * (1 - commonValueImpact * mcf) : _nRows) * 0.6;

		final double numberTuples = g.getNumVals();
		final double tupleSparsity = g.getTupleSparsity();
		final double postScalingCost = (nColsInGroup > 1 && tupleSparsity > 0.4) ? numberTuples * nColsInGroup *
			tupleSparsity * 1.4 : numberTuples * nColsInGroup;

		return preAggregateCost + postScalingCost;
	}

	private double _compressedMultCost(CompressedSizeInfoColGroup g) {
		final int nColsInGroup = g.getColumns().length;
		final double mcf = g.getMostCommonFraction();
		final double preAggregateCost = (mcf > cvThreshold ? _nRows * (1 - commonValueImpact * mcf) : _nRows) * 0.6;
		// final double preAggregateCost = _nRows;

		final double numberTuples = g.getNumVals();
		final double tupleSparsity = g.getTupleSparsity();
		final double postScalingCost = (nColsInGroup > 1 && tupleSparsity > 0.4) ? numberTuples * nColsInGroup *
			tupleSparsity * 1.4 : numberTuples * nColsInGroup;

		return preAggregateCost + postScalingCost;

	}

	private double rightMultCost(CompressedSizeInfoColGroup g) {
		final int nColsInGroup = g.getColumns().length;
		final int numberTuples = g.getNumVals();
		final double tupleSparsity = g.getTupleSparsity();
		final double postScalingCost = (nColsInGroup > 1 && tupleSparsity > 0.4) ? numberTuples * nColsInGroup *
			tupleSparsity * 1.4 : numberTuples * nColsInGroup;
		final double postAllocationCost = 16 * numberTuples;
		return postScalingCost + postAllocationCost;

	}

	private double decompressionCost(CompressedSizeInfoColGroup g) {
		return _nRows * g.getColumns().length * (g.getNumVals() / 64000 + 1);
	}

	private double overlappingDecompressionCost(CompressedSizeInfoColGroup g) {
		final double mcf = g.getMostCommonFraction();
		final double rowsCost = mcf > cvThreshold ? _nRows * (1 - commonValueImpact * mcf) : _nRows;
		// Setting 64 to mark decompression as expensive.
		return rowsCost * 64;
	}

	private static double dictionaryOpsCost(CompressedSizeInfoColGroup g) {
		return g.getColumns().length * g.getNumVals();
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}

	public boolean isDense() {
		return _isDensifying;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("dims(");
		sb.append(_nRows + ",");
		sb.append(_nCols + ") ");
		sb.append("CostVector:[");
		sb.append(_scans + ",");
		sb.append(_decompressions + ",");
		sb.append(_overlappingDecompressions + ",");
		sb.append(_leftMultiplications + ",");
		sb.append(_rightMultiplications + ",");
		sb.append(_compressedMultiplication + ",");
		sb.append(_dictionaryOps + "]");
		sb.append(" Densifying:");
		sb.append(_isDensifying);
		return sb.toString();
	}

}
