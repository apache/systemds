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

import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.sample.HassAndStokes;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.ABitmap.BitmapType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CompressedSizeEstimatorSample extends CompressedSizeEstimator {

	private int[] _sampleRows = null;
	private HashMap<Integer, Double> _solveCache = null;

	/**
	 * CompressedSizeEstimatorSample, samples from the input data and estimates the size of the compressed matrix.
	 * 
	 * @param data         The input data sampled from
	 * @param compSettings The Settings used for the sampling, and compression, contains information such as seed.
	 * @param sampleSize   Size of the sampling used
	 */
	public CompressedSizeEstimatorSample(MatrixBlock data, CompressionSettings compSettings, int sampleSize) {
		super(data, compSettings);

		_sampleRows = getSortedUniformSample(_numRows, sampleSize, _compSettings.seed);

		// Override the _data Matrix block with the sampled matrix block.
		MatrixBlock select = new MatrixBlock(_numRows, 1, false);
		for(int i = 0; i < sampleSize; i++)
			select.quickSetValue(_sampleRows[i], 0, 1);
		_data = _data.removeEmptyOperations(new MatrixBlock(), !_compSettings.transposeInput, true, select);

		// establish estimator-local cache for numeric solve
		_solveCache = new HashMap<>();
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes) {
		int sampleSize = _sampleRows.length;
		int numCols = colIndexes.length;
		int[] sampleRows = _sampleRows;

		// extract statistics from sample
		ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, _data, _compSettings);
		EstimationFactors fact = EstimationFactors.computeSizeEstimationFactors(ubm, false, _numRows, numCols);

		// estimate number of distinct values (incl fixes for anomalies w/ large sample fraction)
		// TODO Replace this with lib matrix/data/LibMatrixCountDistinct
		int totalCardinality = getNumDistinctValues(ubm, _numRows, sampleRows, _solveCache);
		totalCardinality = Math.max(totalCardinality, fact.numVals);
		totalCardinality =  _compSettings.lossy ? Math.min(totalCardinality, numCols * 127) : totalCardinality;
		totalCardinality = Math.min(totalCardinality, _numRows);

		// Number of unseen values
		// int unseenVals = totalCardinality - fact.numVals;

		// Note this numZeros is the count of rows that are all zero.
		int numZeros = ubm.getZeroCounts();

		// estimate number of non-zeros (conservatively round up)
		double C = Math.max(1 - (double) fact.numSingle / sampleSize, (double) sampleSize / _numRows);

		int numNonZeros = (int) Math.ceil(_numRows - (double) _numRows / sampleSize * C * numZeros);
		numNonZeros = Math.max(numNonZeros, totalCardinality); // handle anomaly of zi=0

		// estimate number of segments and number of runs incl correction for
		// empty segments and empty runs (via expected mean of offset value)
		// int numUnseenSeg = (int) (unseenVals * Math.ceil((double) _numRows / BitmapEncoder.BITMAP_BLOCK_SZ / 2));
		int totalNumRuns = ubm.getNumValues() > 0 ? getNumRuns(ubm, sampleSize, _numRows, sampleRows) : 0;

		boolean containsZero = numZeros > 0;

		EstimationFactors totalFacts = new EstimationFactors(numCols, totalCardinality, numNonZeros, totalNumRuns,
			fact.numSingle, _numRows, containsZero, ubm.getType() == BitmapType.Lossy);

		// construct new size info summary
		return new CompressedSizeInfoColGroup(totalFacts, _compSettings.validCompressions);
	}

	private static int getNumDistinctValues(ABitmap ubm, int numRows, int[] sampleRows,
		HashMap<Integer, Double> solveCache) {
		return HassAndStokes.haasAndStokes(ubm, numRows, sampleRows.length, solveCache);
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
	private static int[] getSortedUniformSample(int range, int smplSize, long seed) {
		return UtilFunctions.getSortedSampleIndexes(range, smplSize, seed);
	}

}
