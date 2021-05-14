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

import java.util.HashMap;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.estim.sample.HassAndStokes;
import org.apache.sysds.runtime.compress.lib.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.ABitmap.BitmapType;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CompressedSizeEstimatorSample extends CompressedSizeEstimator {

	private static final int FORCE_TRANSPOSE_ON_SAMPLE_THRESHOLD = 8000;

	private final int[] _sampleRows;
	private HashMap<Integer, Double> _solveCache = null;

	/**
	 * CompressedSizeEstimatorSample, samples from the input data and estimates the size of the compressed matrix.
	 * 
	 * @param data         The input data toSample from
	 * @param compSettings The Settings used for the sampling, and compression, contains information such as seed.
	 * @param sampleRows   The rows sampled
	 * @param transposed   Boolean specifying if the input is already transposed.
	 */
	public CompressedSizeEstimatorSample(MatrixBlock data, CompressionSettings compSettings, int[] sampleRows,
		boolean transposed) {
		super(data, compSettings, transposed);
		_sampleRows = sampleRows;
		_solveCache = new HashMap<>();
		_data = sampleData(data, compSettings, sampleRows, transposed);
	}

	protected MatrixBlock sampleData(MatrixBlock data, CompressionSettings compSettings, int[] sampleRows,
		boolean transposed) {
		MatrixBlock sampledMatrixBlock;
		if(data.isInSparseFormat() && !transposed) {
			sampledMatrixBlock = new MatrixBlock(sampleRows.length, data.getNumColumns(), true);
			SparseRow[] rows = new SparseRow[sampleRows.length];
			SparseBlock in = data.getSparseBlock();
			for(int i = 0; i < sampleRows.length; i++) {
				rows[i] = in.get(sampleRows[i]);
			}
			sampledMatrixBlock.setSparseBlock(new SparseBlockMCSR(rows, false));
			sampledMatrixBlock.recomputeNonZeros();
			_transposed = true;
			sampledMatrixBlock = LibMatrixReorg.transposeInPlace(sampledMatrixBlock, 16);
		}
		else {

			// Override the _data Matrix block with the sampled matrix block.
			MatrixBlock select = (transposed) ? new MatrixBlock(data.getNumColumns(), 1,
				true) : new MatrixBlock(data.getNumRows(), 1, true);
			for(int i = 0; i < sampleRows.length; i++)
				select.appendValue(sampleRows[i], 0, 1);

			sampledMatrixBlock = data.removeEmptyOperations(new MatrixBlock(), !transposed, true, select);

		}

		if(!transposed && sampledMatrixBlock.isInSparseFormat() &&
			sampleRows.length > FORCE_TRANSPOSE_ON_SAMPLE_THRESHOLD) {
			_transposed = true;
			sampledMatrixBlock = LibMatrixReorg.transpose(sampledMatrixBlock,
				new MatrixBlock(sampleRows.length, data.getNumRows(), true), 1);
		}

		if(sampledMatrixBlock.isEmpty())
			throw new DMLCompressionException("Empty sample block");

		return sampledMatrixBlock;

	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes) {
		final int sampleSize = _sampleRows.length;
		// final int numCols = colIndexes.length;

		// extract statistics from sample
		final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, _data, _transposed);
		final EstimationFactors fact = EstimationFactors.computeSizeEstimationFactors(ubm, false, _numRows, colIndexes);
		final int numZerosInSample = ubm.getZeroCounts();
		final boolean lossy = ubm.getType() == BitmapType.Lossy;

		if(numZerosInSample == sampleSize || ubm.getOffsetList() == null) {
			// Since we sample, and this column seems to be empty, we set the return to 1 value detected.
			// aka 1 value, and 1 offset.
			// This makes it more robust in the co coding of Columns
			return new CompressedSizeInfoColGroup(
				new EstimationFactors(colIndexes, 1, 1, _numRows - 1, 2, 1, _numRows, lossy, true, 0, 0),
				_compSettings.validCompressions);
		}

		// estimate number of distinct values (incl fixes for anomalies w/ large sample fraction)
		int totalCardinality = getNumDistinctValues(ubm, _numRows, sampleSize, _solveCache);
		totalCardinality = Math.max(totalCardinality, fact.numVals);
		totalCardinality = lossy ? Math.min(totalCardinality, 256) : totalCardinality;
		totalCardinality = Math.min(totalCardinality, _numRows);

		// Number of unseen values
		// int unseenVals = totalCardinality - fact.numVals;

		// estimate number of non-zeros (conservatively round up)
		final double C = Math.max(1 - (double) fact.numSingle / sampleSize, (double) sampleSize / _numRows);
		final int numNonZeros = Math.max(
			(int) Math.floor(_numRows - (double) (_numRows / sampleSize) * C * numZerosInSample), totalCardinality);

		// estimate number of segments and number of runs incl correction for
		// empty segments and empty runs (via expected mean of offset value)
		// int numUnseenSeg = (int) (unseenVals * Math.ceil((double) _numRows / BitmapEncoder.BITMAP_BLOCK_SZ / 2));
		final int totalNumRuns = _compSettings.validCompressions.contains(CompressionType.RLE) &&
			ubm.getNumValues() > 0 ? getNumRuns(ubm, sampleSize, _numRows, _sampleRows) : 0;

		// Largest instance count ... initiate as the number of zeros.
		int largestInstanceCount = numZerosInSample;
		for(IntArrayList a : ubm.getOffsetList()) {
			if(a.size() > largestInstanceCount)
				largestInstanceCount = a.size();
		}

		final boolean zeroIsMostFrequent = largestInstanceCount == numZerosInSample;

		// Scale largest Instance count to correctly reflect the number of instances.
		largestInstanceCount = (int) (((double) _numRows / sampleSize) * largestInstanceCount);

		EstimationFactors totalFacts = new EstimationFactors(colIndexes, totalCardinality, numNonZeros,
			largestInstanceCount, totalNumRuns, fact.numSingle, _numRows, lossy, zeroIsMostFrequent, fact.tupleSparsity,
			fact.overAllSparsity);

		// construct new size info summary
		return new CompressedSizeInfoColGroup(totalFacts, _compSettings.validCompressions);
	}

	private static int getNumDistinctValues(ABitmap ubm, int numRows, int sampleSize,
		HashMap<Integer, Double> solveCache) {
		return HassAndStokes.haasAndStokes(ubm, numRows, sampleSize, solveCache);
	}

	private static int getNumRuns(ABitmap ubm, int sampleSize, int totalNumRows, int[] sampleRows) {
		int numVals = ubm.getNumValues();
		double numRuns = 0;
		for(int vi = 0; vi < numVals; vi++) {
			int[] offsets = ubm.getOffsetsList(vi).extractValues();
			int offsetsSize = ubm.getNumOffsets(vi);
			double offsetsRatio = ((double) offsetsSize) / sampleSize;
			double avgAdditionalOffsets = offsetsRatio * totalNumRows / sampleSize;
			if(avgAdditionalOffsets < 1) {
				// Ising-Stevens does not hold
				// fall-back to using the expected number of offsets as an upper
				// bound on the number of runs
				numRuns += ((double) offsetsSize) * totalNumRows / sampleSize;
				continue;
			}
			int intervalEnd, intervalSize;
			double additionalOffsets;
			// probability of an index being non-offset in current and previous
			// interval respectively
			double nonOffsetProb, prevNonOffsetProb = 1;
			boolean reachedSampleEnd = false;
			// handling the first interval separately for simplicity
			int intervalStart = -1;
			if(sampleRows[0] == 0) {
				// empty interval
				intervalStart = 0;
			}
			else {
				intervalEnd = sampleRows[0];
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets) * additionalOffsets / intervalSize;
				intervalStart = intervalEnd;
				prevNonOffsetProb = (intervalSize - additionalOffsets) / intervalSize;
			}
			// for handling separators

			int withinSepRun = 0;
			boolean seenNonOffset = false, startedWithOffset = false, endedWithOffset = false;
			int offsetsPtrs = 0;
			for(int ix = 1; ix < sampleSize; ix++) {
				// start of a new separator
				// intervalStart will always be pointing at the current value
				// in the separator block

				if(offsetsPtrs < offsetsSize && offsets[offsetsPtrs] == intervalStart) {
					startedWithOffset = true;
					offsetsPtrs++;
					endedWithOffset = true;
				}
				else {
					seenNonOffset = true;
					endedWithOffset = false;
				}
				while(intervalStart + 1 == sampleRows[ix]) {
					intervalStart = sampleRows[ix];
					if(seenNonOffset) {
						if(offsetsPtrs < offsetsSize && offsets[offsetsPtrs] == intervalStart) {
							withinSepRun = 1;
							offsetsPtrs++;
							endedWithOffset = true;
						}
						else {
							numRuns += withinSepRun;
							withinSepRun = 0;
							endedWithOffset = false;
						}
					}
					else if(offsetsPtrs < offsetsSize && offsets[offsetsPtrs] == intervalStart) {
						offsetsPtrs++;
						endedWithOffset = true;
					}
					else {
						seenNonOffset = true;
						endedWithOffset = false;
					}
					//
					ix++;
					if(ix == sampleSize) {
						// end of sample which searching for a start
						reachedSampleEnd = true;
						break;
					}
				}

				// runs within an interval of unknowns
				if(reachedSampleEnd)
					break;
				intervalEnd = sampleRows[ix];
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets) * additionalOffsets / intervalSize;
				nonOffsetProb = (intervalSize - additionalOffsets) / intervalSize;

				// additional runs resulting from x's on the boundaries of the
				// separators
				// endedWithOffset = findInArray(offsets, intervalStart) != -1;
				if(seenNonOffset) {
					if(startedWithOffset) {
						// add p(y in the previous interval)
						numRuns += prevNonOffsetProb;
					}
					if(endedWithOffset) {
						// add p(y in the current interval)
						numRuns += nonOffsetProb;
					}
				}
				else {
					// add p(y in the previous interval and y in the current
					// interval)
					numRuns += prevNonOffsetProb * nonOffsetProb;
				}
				prevNonOffsetProb = nonOffsetProb;
				intervalStart = intervalEnd;
				// reseting separator variables
				seenNonOffset = startedWithOffset = endedWithOffset = false;
				withinSepRun = 0;

			}
			// last possible interval
			if(intervalStart != totalNumRows - 1) {
				intervalEnd = totalNumRows;
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets) * additionalOffsets / intervalSize;
				nonOffsetProb = (intervalSize - additionalOffsets) / intervalSize;
			}
			else {
				nonOffsetProb = 1;
			}
			// additional runs resulting from x's on the boundaries of the
			// separators
			endedWithOffset = intervalStart == offsets[offsetsSize - 1];
			if(seenNonOffset) {
				if(startedWithOffset) {
					numRuns += prevNonOffsetProb;
				}
				if(endedWithOffset) {
					// add p(y in the current interval)
					numRuns += nonOffsetProb;
				}
			}
			else {
				if(endedWithOffset)
					// add p(y in the previous interval and y in the current
					// interval)
					numRuns += prevNonOffsetProb * nonOffsetProb;
			}
		}
		return (int) Math.min(Math.round(numRuns), Integer.MAX_VALUE);
	}

	/**
	 * Returns a sorted array of n integers, drawn uniformly from the range [0,range).
	 * 
	 * @param range    the range
	 * @param smplSize sample size
	 * @return sorted array of integers
	 */
	protected static int[] getSortedUniformSample(int range, int smplSize, long seed) {
		return UtilFunctions.getSortedSampleIndexes(range, smplSize, seed);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" sampleSize: ");
		sb.append(_sampleRows.length);
		sb.append(" transposed: ");
		sb.append(_transposed);
		sb.append(" cols: ");
		sb.append(_numCols);
		sb.append(" rows: ");
		sb.append(_numRows);
		return sb.toString();
	}
}
