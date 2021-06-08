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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.workload.Op;
import org.apache.sysds.runtime.compress.workload.OpSided;
import org.apache.sysds.runtime.compress.workload.WTreeNode;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;

public class ComputationCostEstimator implements ICostEstimate {

	private final boolean _isCompareAll;
	private final int _nRows;
	// Iteration through each row of decompressed.
	private final int _scans;
	private final int _decompressions;
	// The number of left (rows) multiplied on the matrix.
	private final int _leftMultiplications;
	private final int _rightMultiplications;
	private final int _compressedMultiplication;
	private final int _dictionaryOps;

	/**
	 * A Cost based estimator based on the WTree that is parsed in IPA.
	 * 
	 * @param tree The tree to estimate cost from.
	 */
	private ComputationCostEstimator(int nRows, boolean compareAll, InstructionTypeCounter counts) {
		_nRows = nRows;
		_isCompareAll = compareAll;
		_scans = counts.scans;
		_decompressions = counts.decompressions;
		_leftMultiplications = counts.leftMultiplications;
		_compressedMultiplication = counts.compressedMultiplications;
		_rightMultiplications = counts.rightMultiplications;
		_dictionaryOps = counts.dictionaryOps;
		LOG.error(this);
	}

	public static ComputationCostEstimator create(WTreeRoot tree, int nRows, CompressionSettings cs) {

		InstructionTypeCounter counter = new InstructionTypeCounter();

		for(WTreeNode n : tree.getChildNodes())
			addNode(1, n, counter);

		LOG.error(tree);
		return new ComputationCostEstimator(nRows, false, counter);

	}

	private static void addNode(int count, WTreeNode n, InstructionTypeCounter counter) {

		int mult;
		switch(n.getType()) {
			case IF:
			case FCALL:
			case BASIC_BLOCK:
				mult = 1;
				break;
			case WHILE:
			case FOR:
			case PARFOR:
			default:
				mult = 10;
		}

		for(Op o : n.getOps())
			addOp(count * mult, o, counter);
		for(WTreeNode nc : n.getChildNodes())
			addNode(count * mult, nc, counter);
	}

	private static void addOp(int count, Op o, InstructionTypeCounter counter) {
		if(o instanceof OpSided) {
			OpSided os = (OpSided) o;
			if(os.isLeftMM())
				counter.leftMultiplications += count;
			else if(os.isRightMM())
				counter.rightMultiplications += count;
			else
				counter.compressedMultiplications += count;
			LOG.error("Here");
		}
		else {
			counter.dictionaryOps += count;
		}
	}

	@Override
	public double getUncompressedCost(int nRows, int nCols, int sparsity) {
		throw new NotImplementedException();
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		double cost = 0;
		cost += _scans * scanCost(g);
		cost += _decompressions * decompressionCost(g);
		// 16 is assuming that the left side is 16 rows.
		cost += _leftMultiplications * leftMultCost(g) * 16;
		// 16 is assuming that the right side is 16 rows.
		cost += _rightMultiplications * rightMultCost(g) * 16;
		cost += _dictionaryOps * dictionaryOpsCost(g);
		return cost;
	}

	private double scanCost(CompressedSizeInfoColGroup g) {
		return _nRows;
	}

	private double leftMultCost(CompressedSizeInfoColGroup g) {
		final int nCols = g.getColumns().length;
		final double preAggregateCost = _nRows;

		final int numberTuples = g.getNumVals();
		final double tupleSparsity = g.getTupleSparsity();
		final double postScalingCost = (nCols > 1 && tupleSparsity > 0.4) ? numberTuples * nCols : numberTuples *
			nCols * tupleSparsity;

		return preAggregateCost + postScalingCost;
	}

	private double rightMultCost(CompressedSizeInfoColGroup g) {
		final int nCols = g.getColumns().length;

		final int numberTuples = g.getNumVals();
		final double tupleSparsity = g.getTupleSparsity();
		final double postScalingCost = (nCols > 1 && tupleSparsity > 0.4) ? numberTuples * nCols : numberTuples *
			nCols * tupleSparsity;

		return postScalingCost;
	}

	private double decompressionCost(CompressedSizeInfoColGroup g) {
		return 1000; // ??
	}

	private double dictionaryOpsCost(CompressedSizeInfoColGroup g) {
		return g.getColumns().length * g.getNumVals();
	}

	@Override
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gs) {
		throw new NotImplementedException();
	}

	@Override
	public boolean isCompareAll() {
		return _isCompareAll;
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}

	@Override
	public boolean shouldTryJoin(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append(_nRows + "  ");
		sb.append(_scans + " ");
		sb.append(_decompressions + " ");
		sb.append(_leftMultiplications + " ");
		sb.append(_rightMultiplications + " ");
		sb.append(_compressedMultiplication + " ");
		sb.append(_dictionaryOps + " ");
		return sb.toString();
	}

	protected static class InstructionTypeCounter {
		int scans = 0;
		int decompressions = 0;
		int leftMultiplications = 0;
		int rightMultiplications = 0;
		int compressedMultiplications = 0;
		int dictionaryOps = 1; // base cost is one pass of dictionary

		protected InstructionTypeCounter() {

		}

	}
}
