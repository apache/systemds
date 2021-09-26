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

package org.apache.sysds.runtime.compress.estim;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator.CountDistinctTypes;

/**
 * UltraSparse compressed size estimator (examines entire dataset).
 */
public class CompressedSizeEstimatorUltraSparse extends CompressedSizeEstimator {

	final int nDistinct;

	final EstimationFactors oneColumnFacts;

	private CompressedSizeEstimatorUltraSparse(MatrixBlock data, CompressionSettings compSettings) {
		super(data, compSettings);
		CountDistinctOperator op = new CountDistinctOperator(CountDistinctTypes.COUNT);
		final int _numRows = getNumRows();

		if(LOG.isDebugEnabled()) {
			Timing time = new Timing(true);
			nDistinct = LibMatrixCountDistinct.estimateDistinctValues(data, op);
			LOG.debug("Number Distinct found in entire matrix: " + nDistinct + "  [ms]: " + time.stop());
		}
		else {
			nDistinct = LibMatrixCountDistinct.estimateDistinctValues(data, op);
		}
		final double sparsity = data.getSparsity();
		final int largestOff = (int) ((double) _numRows * (1 - sparsity));
		oneColumnFacts = new EstimationFactors(1, nDistinct, _numRows, largestOff, sparsity);
	}

	protected static CompressedSizeEstimatorUltraSparse create(MatrixBlock data, CompressionSettings compSettings,
		int k) {
		LOG.debug("Using UltraSparse Estimator");
		long oldNNZ = data.getNonZeros();
		if(LOG.isDebugEnabled()) {
			Timing time = new Timing(true);
			data = LibMatrixReorg.transpose(data, new MatrixBlock(data.getNumColumns(), data.getNumRows(), true), k, true);
			LOG.debug("Transposing Input for Ultra Sparse: " + time.stop());
		}
		else
			data = LibMatrixReorg.transpose(data, new MatrixBlock(data.getNumColumns(), data.getNumRows(), true), k, true);

		data.setNonZeros(oldNNZ);
		compSettings.transposed = true;
		return new CompressedSizeEstimatorUltraSparse(data, compSettings);
	}

	@Override
	protected CompressedSizeInfoColGroup[] CompressedSizeInfoColGroup(int clen, int k) {
		// Overwrite parallelization... since we dont want that here.
		return CompressedSizeInfoColGroup(clen);
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int estimate,
		int nrUniqueUpperBound) {
		final int _numRows = getNumRows();
		if(colIndexes.length == 1)
			return new CompressedSizeInfoColGroup(colIndexes, oneColumnFacts, _data.getSparsity());
		else {
			final double sparsity = _data.getSparsity();
			final int nCols = colIndexes.length;
			final int scaledDistinct = (int) Math.min(Math.pow(nDistinct, nCols), Integer.MAX_VALUE);
			final int largestOff = (int) ((double) _numRows * (1 - sparsity * nCols));
			final EstimationFactors facts = new EstimationFactors(nCols, scaledDistinct, _numRows, largestOff, sparsity);
			return new CompressedSizeInfoColGroup(colIndexes, facts, _data.getSparsity());
		}
	}

	@Override
	protected CompressedSizeInfoColGroup estimateJoinCompressedSize(int[] joined, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int joinedMaxDistinct) {
		throw new NotImplementedException();
	}

	@Override
	protected int worstCaseUpperBound(int[] columns) {
		return getNumRows();
	}
}
