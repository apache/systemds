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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorSample extends CompressedSizeEstimator {

	private final MatrixBlock _sample;
	private final HashMap<Integer, Double> _solveCache;
	private final int _k;
	private final int _sampleSize;
	/** Boolean specifying if the sample is in transposed format. */
	private boolean _transposed;

	/**
	 * CompressedSizeEstimatorSample, samples from the input data and estimates the size of the compressed matrix.
	 * 
	 * @param data       The input data toSample from
	 * @param cs         The Settings used for the sampling, and compression, contains information such as seed.
	 * @param sampleSize The size to sample from the data.
	 * @param k          The parallelization degree allowed.
	 */
	protected CompressedSizeEstimatorSample(MatrixBlock data, CompressionSettings cs, int sampleSize, int k) {
		super(data, cs);
		_k = k;
		_sampleSize = sampleSize;
		_transposed = _cs.transposed;
		if(LOG.isDebugEnabled()) {
			Timing time = new Timing(true);
			_sample = sampleData(sampleSize);
			LOG.debug("Sampling time: " + time.stop());
		}
		else
			_sample = sampleData(sampleSize);

		_solveCache = new HashMap<>();
	}

	public MatrixBlock getSample() {
		return _sample;
	}

	public final int getSampleSize() {
		return _sampleSize;
	}

	@Override
	public CompressedSizeInfoColGroup estimateCompressedColGroupSize(int[] colIndexes, int estimate,
		int nrUniqueUpperBound) {

		// extract statistics from sample
		final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, _sample, _transposed, estimate, false);
		final EstimationFactors sampleFacts = EstimationFactors.computeSizeEstimationFactors(ubm, _sampleSize, false,
			colIndexes);
		final AMapToData map = MapToFactory.create(_sampleSize, ubm);

		// result scaled
		final EstimationFactors em = estimateCompressionFactors(sampleFacts, map, colIndexes, nrUniqueUpperBound);

		// LOG.error("Sample vs Scaled:\n" + sampleFacts + "\n" + em + "\n");

		return new CompressedSizeInfoColGroup(colIndexes, em, _cs.validCompressions, map);
	}

	@Override
	protected int worstCaseUpperBound(int[] columns) {
		if(getNumColumns() == columns.length)
			return Math.min(getNumRows(), (int) _data.getNonZeros());
		return getNumRows();
	}

	@Override
	protected CompressedSizeInfoColGroup estimateJoinCompressedSize(int[] joined, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int joinedMaxDistinct) {
		if((long) g1.getNumVals() * g2.getNumVals() > (long) Integer.MAX_VALUE)
			return null;

		final AMapToData map = MapToFactory.join(g1.getMap(), g2.getMap());
		final EstimationFactors sampleFacts = EstimationFactors.computeSizeEstimation(joined, map,
			_cs.validCompressions.contains(CompressionType.RLE), map.size(), false);

		// result facts
		final EstimationFactors em = estimateCompressionFactors(sampleFacts, map, joined, joinedMaxDistinct);

		// LOG.error("Sample vs Scaled Join:\n" + sampleFacts + "\n" + em + "\n");

		return new CompressedSizeInfoColGroup(joined, em, _cs.validCompressions, map);
	}

	private EstimationFactors estimateCompressionFactors(EstimationFactors sampleFacts, AMapToData map, int[] colIndexes,
		int nrUniqueUpperBound) {
		final int numRows = getNumRows();
		if(map == null || sampleFacts == null) {
			// This column's sample is empty, try to return some best estimate based on the matrix metadata.
			final int nCol = colIndexes.length;
			if(_data.isEmpty()) // The entire matrix was empty therefore return empty statistics for this column Group.
				return new EstimationFactors(colIndexes.length, 0, 0, numRows, null, 0, 0, numRows, false, true, 0.0, 0.0);
			final int largestInstanceCount = numRows - 1;
			return new EstimationFactors(colIndexes.length, 1, 1, largestInstanceCount, null, 2, 1, numRows, false, true,
				(double) 1 / numRows, (double) 1 / nCol);
		}
		else {
			final int numZerosInSample = sampleFacts.numRows - sampleFacts.numOffs;
			// estimate number of non-zeros (conservatively round up)
			final double scalingFactor = (double) numRows / _sampleSize;

			final int numOffs = calculateOffs(sampleFacts, _sampleSize, numRows, scalingFactor, numZerosInSample);

			final int totalCardinality = getEstimatedDistinctCount(sampleFacts.frequencies, nrUniqueUpperBound, numOffs,
				sampleFacts.numOffs);

			final int totalNumRuns = getNumRuns(map, sampleFacts.numVals, _sampleSize, numRows);

			final int largestInstanceCount = Math.min(numRows - totalCardinality + 1,
				(int) Math.floor(sampleFacts.largestOff * scalingFactor));

			final double overallSparsity = calculateSparsity(colIndexes, scalingFactor, sampleFacts.overAllSparsity);

			return new EstimationFactors(colIndexes.length, totalCardinality, numOffs, largestInstanceCount,
				sampleFacts.frequencies, totalNumRuns, sampleFacts.numSingle, numRows, sampleFacts.lossy,
				sampleFacts.zeroIsMostFrequent, overallSparsity, sampleFacts.tupleSparsity);
		}
	}

	private int calculateOffs(EstimationFactors sampleFacts, int sampleSize, int numRows, double scalingFactor,
		int numZerosInSample) {
		final int numCols = getNumColumns();
		if(numCols == 1)
			return (int) _data.getNonZeros();
		else {
			final double C = Math.max(1 - (double) sampleFacts.numSingle / sampleSize, (double) sampleSize / numRows);
			return (int) Math.ceil(numRows - scalingFactor * C * numZerosInSample);
		}
	}

	private double calculateSparsity(int[] colIndexes, double scalingFactor, double sampleValue) {
		if(colIndexes.length == getNumColumns())
			return _data.getSparsity();
		else if(_cs.transposed && _data.isInSparseFormat()) {
			// Use exact if possible
			double nnzCount = 0;
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < colIndexes.length; i++)
				nnzCount += sb.get(i).size();
			return nnzCount / ((double) getNumRows() * colIndexes.length);
		}
		else if(_transposed && _sample.isInSparseFormat()) {
			// Fallback to the sample if original is not transposed
			double nnzCount = 0;
			SparseBlock sb = _sample.getSparseBlock();
			for(int i = 0; i < colIndexes.length; i++)
				if(!sb.isEmpty(i))
					nnzCount += (double) sb.get(i).size() * scalingFactor;

			// add one to make sure that Uncompressed columns are considered as containing at least one value.
			if(nnzCount == 0)
				nnzCount += 1;
			return nnzCount / ((double) getNumRows() * colIndexes.length);
		}
		else
			// if all others aren't available use the samples value.
			return sampleValue;
	}

	private int getEstimatedDistinctCount(int[] freq, int upperBound, int numOffs, int numOffsInSample) {
		final int est = SampleEstimatorFactory.distinctCount(freq, numOffs, numOffsInSample, _cs.estimationType,
			_solveCache);
		return Math.min(est, upperBound);
	}

	private int getNumRuns(AMapToData map, int numVals, int sampleSize, int totalNumRows) {
		// estimate number of segments and number of runs incl correction for
		// empty segments and empty runs (via expected mean of offset value)
		// int numUnseenSeg = (int) (unseenVals * Math.ceil((double) _numRows / BitmapEncoder.BITMAP_BLOCK_SZ / 2));
		return _cs.validCompressions.contains(CompressionType.RLE) && numVals > 0 ? getNumRuns(map, sampleSize,
			totalNumRows) : 0;
	}

	// Fix getNumRuns when adding RLE back.
	@SuppressWarnings("unused")
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

	private static int getNumRuns(AMapToData map, int sampleSize, int totalNumRows) {
		throw new NotImplementedException("Not Supported ever since the ubm was replaced by the map");
	}

	private static int[] getSortedSample(int range, int sampleSize, long seed, int k) {
		// set meta data and allocate dense block
		final int[] a = new int[sampleSize];
		seed = (seed == -1 ? System.nanoTime() : seed);

		Random r = new Random(seed);
		// reservoir sampling
		for(int i = 0; i < sampleSize; i++)
			a[i] = i;

		for(int i = sampleSize; i < range; i++)
			if(r.nextInt(i) < sampleSize)
				a[r.nextInt(sampleSize)] = i;

		if(range / 100 < sampleSize) {
			// randomize the sample (Algorithm P from Knuth's ACP)
			// needed especially when the difference between range and sampleSize is small)
			for(int i = 0; i < sampleSize - 1; i++) {
				// generate index in i <= j < sampleSize
				int j = r.nextInt(sampleSize - i) + i;
				// swap i^th and j^th entry
				int tmp = a[i];
				a[i] = a[j];
				a[j] = tmp;
			}
		}

		// Sort the sample
		if(k > 1)
			Arrays.parallelSort(a);
		else
			Arrays.sort(a);
		return a;
	}

	private MatrixBlock sampleData(int sampleSize) {
		Timing time = new Timing(true);
		final int[] sampleRows = CompressedSizeEstimatorSample.getSortedSample(getNumRows(), sampleSize, _cs.seed, _k);
		LOG.debug("sampleRow:" + time.stop());

		MatrixBlock sampledMatrixBlock;
		if(!_cs.transposed) {
			if(_data.isInSparseFormat())
				sampledMatrixBlock = sparseNotTransposedSamplePath(sampleRows);
			else
				sampledMatrixBlock = denseSamplePath(sampleRows);
		}
		else
			sampledMatrixBlock = defaultSlowSamplingPath(sampleRows);

		if(sampledMatrixBlock.isEmpty())
			return null;
		else
			return sampledMatrixBlock;
	}

	private MatrixBlock sparseNotTransposedSamplePath(int[] sampleRows) {
		MatrixBlock res = new MatrixBlock(sampleRows.length, _data.getNumColumns(), true);
		SparseRow[] rows = new SparseRow[sampleRows.length];
		SparseBlock in = _data.getSparseBlock();
		for(int i = 0; i < sampleRows.length; i++)
			rows[i] = in.get(sampleRows[i]);

		res.setSparseBlock(new SparseBlockMCSR(rows, false));
		res.recomputeNonZeros();
		_transposed = true;
		res = LibMatrixReorg.transposeInPlace(res, _k);

		return res;
	}

	private MatrixBlock defaultSlowSamplingPath(int[] sampleRows) {
		MatrixBlock select = (_cs.transposed) ? new MatrixBlock(_data.getNumColumns(), 1,
			false) : new MatrixBlock(_data.getNumRows(), 1, false);
		for(int i = 0; i < sampleRows.length; i++)
			select.appendValue(sampleRows[i], 0, 1);
		MatrixBlock ret = _data.removeEmptyOperations(new MatrixBlock(), !_cs.transposed, true, select);
		return ret;
	}

	private MatrixBlock denseSamplePath(int[] sampleRows) {
		final int sampleSize = sampleRows.length;
		final double sampleRatio = _cs.transposed ? (double) _data.getNumColumns() /
			sampleSize : (double) _data.getNumRows() / sampleSize;
		final long inputNonZeros = _data.getNonZeros();
		final long estimatedNonZerosInSample = (long) Math.ceil((double) inputNonZeros / sampleRatio);
		final int resRows = _cs.transposed ? _data.getNumRows() : _data.getNumColumns();
		final long nCellsInSample = (long) sampleSize * resRows;
		final boolean shouldBeSparseSample = 0.4 > (double) estimatedNonZerosInSample / nCellsInSample;
		MatrixBlock res = new MatrixBlock(resRows, sampleSize, shouldBeSparseSample);
		res.allocateBlock();

		final DenseBlock inb = _data.getDenseBlock();
		if(res.isInSparseFormat()) {
			final SparseBlock resb = res.getSparseBlock();
			if(resb instanceof SparseBlockMCSR) {
				final SparseBlockMCSR resbmcsr = (SparseBlockMCSR) resb;
				final int estimatedNrDoublesEachRow = (int) Math.max(4, Math.ceil(estimatedNonZerosInSample / sampleSize));
				for(int col = 0; col < resRows; col++)
					resbmcsr.allocate(col, estimatedNrDoublesEachRow);

				for(int row = 0; row < sampleSize; row++) {
					final int inRow = sampleRows[row];
					final double[] inBlockV = inb.values(inRow);
					final int offIn = inb.pos(inRow);
					for(int col = 0; col < resRows; col++) {
						final SparseRow srow = resbmcsr.get(col);
						srow.append(row, inBlockV[offIn + col]);
					}
				}
			}
			else
				throw new NotImplementedException(
					"Not Implemented support for dense sample into sparse: " + resb.getClass().getSimpleName());

		}
		else {
			final DenseBlock resb = res.getDenseBlock();
			for(int row = 0; row < sampleSize; row++) {
				final int inRow = sampleRows[row];
				final double[] inBlockV = inb.values(inRow);
				final int offIn = inb.pos(inRow);
				for(int col = 0; col < resRows; col++) {
					final double[] blockV = resb.values(col);
					blockV[col * sampleSize + row] = inBlockV[offIn + col];
				}
			}
		}
		res.setNonZeros(estimatedNonZerosInSample);
		_transposed = true;

		return res;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(" sampleSize: ");
		sb.append(getSampleSize());
		sb.append(" transposed: ");
		sb.append(_transposed);
		return sb.toString();
	}
}
