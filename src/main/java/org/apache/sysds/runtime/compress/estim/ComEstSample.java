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
import java.util.Random;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.Timing;

/**
 * Estimate compression size based on subsample of data.
 */
public class ComEstSample extends AComEst {

	/** Sample extracted from the input data */
	protected final MatrixBlock _sample;
	/** Parallelization degree */
	protected final int _k;
	/** Sample size */
	protected final int _sampleSize;
	/** Boolean specifying if the sample is in transposed format. */
	protected boolean _transposed;

	public ComEstSample(MatrixBlock sample, CompressionSettings cs, MatrixBlock full, int k) {
		super(full, cs);
		_k = k;
		_transposed = cs.transposed;
		_sample = sample;
		_sampleSize = sample.getNumRows();

	}

	/**
	 * CompressedSizeEstimatorSample, samples from the input data and estimates the size of the compressed matrix.
	 * 
	 * @param data       The input data toSample from
	 * @param cs         The Settings used for the sampling, and compression, contains information such as seed.
	 * @param sampleSize The size to sample from the data.
	 * @param k          The parallelization degree allowed.
	 */
	public ComEstSample(MatrixBlock data, CompressionSettings cs, int sampleSize, int k) {
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

	}

	@Override
	public CompressedSizeInfoColGroup getColGroupInfo(IColIndex colIndexes, int estimate, int maxDistinct) {
		if(_data.isEmpty() || (nnzCols != null && colIndexes.size() == 1 && nnzCols[colIndexes.get(0)] == 0) ||
			(_cs.transposed && colIndexes.size() == 1 && _data.isInSparseFormat() &&
				_data.getSparseBlock().isEmpty(colIndexes.get(0))))
			return new CompressedSizeInfoColGroup(colIndexes, getNumRows(), CompressionType.EMPTY);

		final IEncode map = EncodingFactory.createFromMatrixBlock(_sample, _transposed, colIndexes);
		return extractInfo(map, colIndexes, maxDistinct);
	}

	@Override
	public CompressedSizeInfoColGroup getDeltaColGroupInfo(IColIndex colIndexes, int estimate, int maxDistinct) {
		// Don't use sample when doing estimation of delta encoding, instead we read from the start of the matrix until
		// sample size. This guarantees that the delta values are actually represented in the full compression
		final IEncode map = EncodingFactory.createFromMatrixBlockDelta(_data, _transposed, colIndexes, _sampleSize);
		return extractInfo(map, colIndexes, maxDistinct);
	}

	@Override
	protected int worstCaseUpperBound(IColIndex columns) {
		if(getNumColumns() == columns.size())
			return Math.min(getNumRows(), (int) Math.min(_data.getNonZeros(), Integer.MAX_VALUE));
		return getNumRows();
	}

	@Override
	protected CompressedSizeInfoColGroup combine(IColIndex combinedColumns, CompressedSizeInfoColGroup g1,
		CompressedSizeInfoColGroup g2, int maxDistinct) {
		try {
			final IEncode map = g1.getMap().combine(g2.getMap());
			return extractInfo(map, combinedColumns, maxDistinct);
		}
		catch(Exception e) {

			String s1 = g1.toString();
			if(s1.length() > 1000)
				s1 = s1.substring(0, 1000);

			String s2 = g2.toString();
			if(s2.length() > 1000)
				s2 = s2.substring(0, 1000);

			throw new DMLCompressionException("Failed to combine :\n" + s1 + "\n\n" + s2, e);
		}
	}

	private CompressedSizeInfoColGroup extractInfo(IEncode map, IColIndex colIndexes, int maxDistinct) {
		try {
			final double spar = _data.getSparsity();
			final EstimationFactors sampleFacts = map.extractFacts(_sampleSize, spar, spar, _cs);
			final EstimationFactors em = scaleFactors(sampleFacts, colIndexes, maxDistinct, map.isDense());
			return new CompressedSizeInfoColGroup(colIndexes, em, _cs.validCompressions, map);
		}
		catch(Exception e) {
			String ms = map.toString();
			if(ms.length() > 1000)
				ms = ms.substring(0, 1000);
			throw new DMLCompressionException("Failed to extract info: \n" + ms, e);
		}
	}

	private EstimationFactors scaleFactors(EstimationFactors sampleFacts, IColIndex colIndexes, int maxDistinct,
		boolean dense) {
		try {

			final int numRows = getNumRows();
			final int nCol = colIndexes.size();

			final double scalingFactor = (double) numRows / _sampleSize;

			final long nnz = calculateNNZ(colIndexes, scalingFactor);
			final int numOffs = calculateOffs(sampleFacts, numRows, scalingFactor, colIndexes, (int) nnz);
			final int estDistinct = distinctCountScale(sampleFacts, numOffs, numRows, maxDistinct, dense, nCol);
			// if(estDistinct < sampleFacts.numVals)
			// throw new DMLCompressionException("Failed estimating distinct: " + estDistinct + " should have been above "
			// + sampleFacts.numVals + "\n" + Arrays.toString(sampleFacts.frequencies));

			// calculate the largest instance count.
			final int maxLargestInstanceCount = numRows - estDistinct + 1;
			final int scaledLargestInstanceCount = sampleFacts.largestOff < 0 ? numOffs /
				estDistinct : (int) Math.floor(sampleFacts.largestOff * scalingFactor);
			final int mostFrequentOffsetCount = Math.max(Math.min(maxLargestInstanceCount, scaledLargestInstanceCount),
				numRows - numOffs);

			final double overallSparsity = calculateSparsity(colIndexes, nnz, scalingFactor, sampleFacts.overAllSparsity);
			// For robustness safety add 10 percent more tuple sparsity
			final double tupleSparsity = Math.min(overallSparsity * 1.3, 1.0); // increase sparsity by 30%.
			if(_cs.isRLEAllowed()) {
				final int scaledRuns = Math.max(estDistinct,
					calculateRuns(sampleFacts, scalingFactor, numOffs, estDistinct));
				return new EstimationFactors(estDistinct, numOffs, mostFrequentOffsetCount, sampleFacts.frequencies,
					sampleFacts.numSingle, numRows, scaledRuns, sampleFacts.lossy, sampleFacts.zeroIsMostFrequent,
					overallSparsity, tupleSparsity);
			}
			else
				return new EstimationFactors(estDistinct, numOffs, mostFrequentOffsetCount, sampleFacts.frequencies,
					sampleFacts.numSingle, numRows, sampleFacts.lossy, sampleFacts.zeroIsMostFrequent, overallSparsity,
					tupleSparsity);
		}
		catch(Exception e) {
			throw new RuntimeException(colIndexes.toString(), e);
		}
	}

	private int distinctCountScale(EstimationFactors sampleFacts, int numOffs, int numRows, int maxDistinct,
		boolean dense, int nCol) {
		// the frequencies of non empty entries.
		final int[] freq = sampleFacts.frequencies;
		if(freq == null || freq.length == 0)
			return numOffs; // very aggressive number of distinct
		maxDistinct = Math.max(maxDistinct, sampleFacts.numVals);
		// sampled size is smaller than actual if there was empty rows.
		// and the more we can reduce this value the more accurate the estimation will become.
		final int sampledSize = sampleFacts.numOffs;
		int est = SampleEstimatorFactory.distinctCount(freq, dense ? numRows : numOffs, sampledSize, _cs.estimationType);
		if(est > 10000)
			est += est * 0.5;
		if(nCol > 4 && est > 100) // Increase estimate if we get into many columns cocoding to be safe
			est += ((double) est) * ((double) nCol) / 10;
		// Bound the estimate with the maxDistinct.
		return Math.max(Math.min(est, Math.min(maxDistinct, numOffs)), 1);
	}

	private int calculateOffs(EstimationFactors sampleFacts, int numRows, double scalingFactor, IColIndex colIndexes,
		int nnz) {

		if(getNumColumns() == 1)
			return nnz;
		else if(nnzCols != null) {
			if(colIndexes.size() == 1)
				return nnzCols[colIndexes.get(0)];
			else {
				final int emptyTuples = sampleFacts.numRows - sampleFacts.numOffs;
				final int estOffs = numRows - (int) Math.floor(emptyTuples * scalingFactor);
				return Math.min(nnz, estOffs);
			}
		}
		else {
			final int emptyTuples = sampleFacts.numRows - sampleFacts.numOffs;
			return numRows - (int) Math.floor(emptyTuples * scalingFactor);
		}
	}

	private int calculateRuns(EstimationFactors sampleFacts, double scalingFactor, int estOffs, int estDistinct) {
		// naive approach.
		final double nRunsInSample = sampleFacts.numRuns;
		double numRuns = 0;

		// process frequency maps.
		// if(sampleFacts.frequencies != null) {
		// for(int freq : sampleFacts.frequencies) {
		// double dFreq = freq;
		// double offsetRatio = dFreq / _sampleSize;
		// double avgOffsetsThisValue = dFreq * scalingFactor;
		// if(offsetRatio < 1) {
		// // Assuming uniform distribution and this value is very rare.
		// // Assume worst case of all unique runs for this value.
		// numRuns += avgOffsetsThisValue;
		// }
		// else {
		// // In the case where we know guaranteed runs because ratio is above 1 we slack the conditions.
		// // saying we have runs based on the ratio of a specific offset is present.
		// numRuns += dFreq / offsetRatio * scalingFactor;
		// }
		// }
		// }

		// numRuns = Math.max(numRuns, nRunsInSample * scalingFactor); // minimum estimate sample runs scaling up.
		double sampleToRunRatio = nRunsInSample / sampleFacts.numVals;
		double sampleSizeToRunRatio = nRunsInSample / _sampleSize;
		numRuns = (sampleToRunRatio <= 1.1 && sampleSizeToRunRatio < 0.5) ? // decide estimation model
			sampleToRunRatio * estDistinct : // run per value in sample scale to larger
			nRunsInSample * scalingFactor; // simply scale num runs

		// With an estimated num runs, we now bound it.
		numRuns = Math.min(numRuns, estOffs); // max number of runs equal to estimated offsets
		numRuns = Math.max(numRuns, estDistinct); // minimum number of distinct
		numRuns = Math.min(Integer.MAX_VALUE, Math.ceil(numRuns));

		return (int) numRuns;
	}

	private double calculateSparsity(IColIndex colIndexes, long nnz, double scalingFactor, double sampleValue) {
		if(colIndexes.size() == getNumColumns())
			return _data.getSparsity();
		else if(nnzCols != null || (_cs.transposed && _data.isInSparseFormat()) ||
			(_transposed && _sample.isInSparseFormat()))
			return (double) nnz / (getNumRows() * colIndexes.size());
		else if(_sample.isEmpty())
			// Make a semi safe bet of using the data input sparsity if the sample was empty.
			return _data.getSparsity();
		else
			return sampleValue;
	}

	private long calculateNNZ(IColIndex colIndexes, double scalingFactor) {
		if(colIndexes.size() == getNumColumns())
			return _data.getNonZeros();
		else if(_cs.transposed && _data.isInSparseFormat()) {
			// Use exact if possible
			long nnzCount = 0;
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < colIndexes.size(); i++)
				nnzCount += sb.get(i).size();
			return nnzCount;
		}
		else if(nnzCols != null) {
			long nnz = 0;
			for(int i = 0; i < colIndexes.size(); i++)
				nnz += nnzCols[colIndexes.get(i)];
			return nnz;
		}
		else if(_sample.isEmpty())
			return 0;
		else if(_transposed && _sample.isInSparseFormat()) {
			// Fallback to the sample if original is not transposed
			long nnzCount = 0;
			SparseBlock sb = _sample.getSparseBlock();
			for(int i = 0; i < colIndexes.size(); i++)
				if(!sb.isEmpty(i))
					nnzCount += sb.get(i).size() * scalingFactor;

			// add one to make sure that Uncompressed columns are considered as containing at least one value.
			if(nnzCount == 0)
				nnzCount += colIndexes.size();
			return nnzCount;
		}
		else
			// if all others aren't available use the samples value.
			return _sample.getNonZeros();
	}

	public static int[] getSortedSample(int range, int sampleSize, long seed, int k) {
		// set meta data and allocate dense block
		final int[] a = new int[sampleSize];

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

		final int[] sampleRows = ComEstSample.getSortedSample(getNumRows(), sampleSize, _cs.seed, _k);
		MatrixBlock sampledMatrixBlock;
		if(!_cs.transposed) {
			if(_data.isInSparseFormat())
				sampledMatrixBlock = sparseNotTransposedSamplePath(sampleRows);
			else
				sampledMatrixBlock = denseSamplePath(sampleRows);
		}
		else
			sampledMatrixBlock = defaultSlowSamplingPath(sampleRows);

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
		final long estimatedNonZerosInSample = (long) Math.ceil(inputNonZeros / sampleRatio);
		final int resRows = _cs.transposed ? _data.getNumRows() : _data.getNumColumns();
		final long nCellsInSample = (long) sampleSize * resRows;
		final boolean shouldBeSparseSample = 0.4 > (double) estimatedNonZerosInSample / nCellsInSample;
		MatrixBlock res = new MatrixBlock(resRows, sampleSize, shouldBeSparseSample);
		res.allocateBlock();

		final DenseBlock inb = _data.getDenseBlock();
		if(res.isInSparseFormat()) {
			final SparseBlock resb = res.getSparseBlock();
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
		sb.append(_sampleSize);
		sb.append(" transposed: ");
		sb.append(_transposed);
		return sb.toString();
	}
}
