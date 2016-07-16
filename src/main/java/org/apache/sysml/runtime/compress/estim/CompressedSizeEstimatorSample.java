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

package org.apache.sysml.runtime.compress.estim;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.compress.BitmapEncoder;
import org.apache.sysml.runtime.compress.ReaderColumnSelection;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.compress.ReaderColumnSelectionDense;
import org.apache.sysml.runtime.compress.ReaderColumnSelectionDenseSample;
import org.apache.sysml.runtime.compress.ReaderColumnSelectionSparse;
import org.apache.sysml.runtime.compress.UncompressedBitmap;
import org.apache.sysml.runtime.compress.utils.DblArray;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorSample extends CompressedSizeEstimator 
{
	private static final boolean CORRECT_NONZERO_ESTIMATE = false; //TODO enable for production
	private final static double SHLOSSER_JACKKNIFE_ALPHA = 0.975;
	public static final float HAAS_AND_STOKES_ALPHA1 = 0.9F; //0.9 recommended in paper
	public static final float HAAS_AND_STOKES_ALPHA2 = 30F; //30 recommended in paper
	public static final float HAAS_AND_STOKES_UJ2A_C = 50; //50 recommend in paper

	private int[] _sampleRows = null;
	private RandomDataGenerator _rng = null;
	private int _numRows = -1;
	
	/**
	 * 
	 * @param data
	 * @param sampleRows
	 */
	public CompressedSizeEstimatorSample(MatrixBlock data, int[] sampleRows) {
		super(data);
		_sampleRows = sampleRows;
		_rng = new RandomDataGenerator();
		_numRows = CompressedMatrixBlock.TRANSPOSE_INPUT ? 
				_data.getNumColumns() : _data.getNumRows();
	}

	/**
	 * 
	 * @param mb
	 * @param sampleSize
	 */
	public CompressedSizeEstimatorSample(MatrixBlock mb, int sampleSize) {
		this(mb, null);
		_sampleRows = getSortedUniformSample(_numRows, sampleSize);
	}

	/**
	 * 
	 * @param sampleRows, assumed to be sorted
	 */
	public void setSampleRows(int[] sampleRows) {
		_sampleRows = sampleRows;
	}

	/**
	 * 
	 * @param sampleSize
	 */
	public void resampleRows(int sampleSize) {
		_sampleRows = getSortedUniformSample(_numRows, sampleSize);
	}

	@Override
	public CompressedSizeInfo estimateCompressedColGroupSize(int[] colIndexes) 
	{
		//extract statistics from sample
		UncompressedBitmap ubm = BitmapEncoder.extractBitmapFromSample(
				colIndexes, _data, _sampleRows);
		SizeEstimationFactors fact = computeSizeEstimationFactors(ubm, false);

		//estimate number of distinct values 
		int totalCardinality = getNumDistinctValues(colIndexes);
		totalCardinality = Math.max(totalCardinality, fact.numVals); //fix anomalies w/ large sample fraction
		totalCardinality = Math.min(totalCardinality, _numRows); //fix anomalies w/ large sample fraction
		
		//estimate unseen values
		// each unseen is assumed to occur only once (it did not show up in the sample because it is rare)
		int unseen = Math.max(0, totalCardinality - fact.numVals);
		int sampleSize = _sampleRows.length;
		
		//estimate number of offsets
		double sparsity = OptimizerUtils.getSparsity(
				_data.getNumRows(), _data.getNumColumns(), _data.getNonZeros());
		
		// expected value given that we don't store the zero values
		float totalNumOffs = (float) (_numRows * (1 - Math.pow(1 - sparsity,colIndexes.length)));		
		if( CORRECT_NONZERO_ESTIMATE ) {
			long numZeros = sampleSize - fact.numOffs;
			float C = Math.max(1-(float)fact.numSingle/sampleSize, (float)sampleSize/_numRows); 
			totalNumOffs = _numRows - ((numZeros>0)? (float)_numRows/sampleSize*C*numZeros : 0);
		}
		
		// For a single offset, the number of blocks depends on the value of
		// that offset. small offsets (first group of rows in the matrix)
		// require a small number of blocks and large offsets (last group of
		// rows) require a large number of blocks. The unseen offsets are
		// distributed over the entire offset range. A reasonable and fast
		// estimate for the number of blocks is to use the arithmetic mean of
		// the number of blocks used for the first index (=1) and that of the
		// last index.
		int numUnseenSeg = Math.round(unseen
				* (2.0f * BitmapEncoder.BITMAP_BLOCK_SZ + _numRows) / 2
				/ BitmapEncoder.BITMAP_BLOCK_SZ);
		int totalNumSeg = fact.numSegs + numUnseenSeg;
		int totalNumRuns = getNumRuns(ubm, sampleSize, _numRows) + unseen;

		//construct new size info summary
		return new CompressedSizeInfo(totalCardinality,
				getRLESize(totalCardinality, totalNumRuns, colIndexes.length),
				getOLESize(totalCardinality, totalNumOffs, totalNumSeg, colIndexes.length));
	}

	@Override
	public CompressedSizeInfo estimateCompressedColGroupSize(UncompressedBitmap ubm) 
	{
		//compute size estimation factors
		SizeEstimationFactors fact = computeSizeEstimationFactors(ubm, true);
		
		//construct new size info summary
		return new CompressedSizeInfo(fact.numVals,
				getRLESize(fact.numVals, fact.numRuns, ubm.getNumColumns()),
				getOLESize(fact.numVals, fact.numOffs, fact.numSegs, ubm.getNumColumns()));
	}
	
	/**
	 * 
	 * @param colIndexes
	 * @return
	 */
	private int getNumDistinctValues(int[] colIndexes) {
		return haasAndStokes(colIndexes);
	}

	/**
	 * 
	 * @param sampleUncompressedBitmap
	 * @param sampleSize
	 * @param totalNumRows
	 * @return
	 */
	private int getNumRuns(UncompressedBitmap sampleUncompressedBitmap,
			int sampleSize, int totalNumRows) {
		int numVals = sampleUncompressedBitmap.getNumValues();
		// all values in the sample are zeros
		if (numVals == 0)
			return 0;
		float numRuns = 0;
		for (int vi = 0; vi < numVals; vi++) {
			int[] offsets = sampleUncompressedBitmap.getOffsetsList(vi);
			float offsetsRatio = ((float) offsets.length) / sampleSize;
			float avgAdditionalOffsets = offsetsRatio * totalNumRows
					/ sampleSize;
			if (avgAdditionalOffsets < 1) {
				// Ising-Stevens does not hold
				// fall-back to using the expected number of offsets as an upper
				// bound on the number of runs
				numRuns += ((float) offsets.length) * totalNumRows / sampleSize;
				continue;
			}
			int intervalEnd, intervalSize;
			float additionalOffsets;
			// probability of an index being non-offset in current and previous
			// interval respectively
			float nonOffsetProb, prevNonOffsetProb = 1;
			boolean reachedSampleEnd = false;
			// handling the first interval separately for simplicity
			int intervalStart = -1;
			if (_sampleRows[0] == 0) {
				// empty interval
				intervalStart = 0;
			} else {
				intervalEnd = _sampleRows[0];
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets)
						* additionalOffsets / intervalSize;
				intervalStart = intervalEnd;
				prevNonOffsetProb = (intervalSize - additionalOffsets)
						/ intervalSize;
			}
			// for handling separators

			int withinSepRun = 0;
			boolean seenNonOffset = false, startedWithOffset = false, endedWithOffset = false;
			int offsetsPtrs = 0;
			for (int ix = 1; ix < sampleSize; ix++) {
				// start of a new separator
				// intervalStart will always be pointing at the current value
				// in the separator block

				if (offsetsPtrs < offsets.length
						&& offsets[offsetsPtrs] == intervalStart) {
					startedWithOffset = true;
					offsetsPtrs++;
					endedWithOffset = true;
				} else {
					seenNonOffset = true;
					endedWithOffset = false;
				}
				while (intervalStart + 1 == _sampleRows[ix]) {
					intervalStart = _sampleRows[ix];
					if (seenNonOffset) {
						if (offsetsPtrs < offsets.length
								&& offsets[offsetsPtrs] == intervalStart) {
							withinSepRun = 1;
							offsetsPtrs++;
							endedWithOffset = true;
						} else {
							numRuns += withinSepRun;
							withinSepRun = 0;
							endedWithOffset = false;
						}
					} else if (offsetsPtrs < offsets.length
							&& offsets[offsetsPtrs] == intervalStart) {
						offsetsPtrs++;
						endedWithOffset = true;
					} else {
						seenNonOffset = true;
						endedWithOffset = false;
					}
					//
					ix++;
					if (ix == sampleSize) {
						// end of sample which searching for a start
						reachedSampleEnd = true;
						break;
					}
				}

				// runs within an interval of unknowns
				if (reachedSampleEnd)
					break;
				intervalEnd = _sampleRows[ix];
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets)
						* additionalOffsets / intervalSize;
				nonOffsetProb = (intervalSize - additionalOffsets)
						/ intervalSize;

				// additional runs resulting from x's on the boundaries of the
				// separators
				// endedWithOffset = findInArray(offsets, intervalStart) != -1;
				if (seenNonOffset) {
					if (startedWithOffset) {
						// add p(y in the previous interval)
						numRuns += prevNonOffsetProb;
					}
					if (endedWithOffset) {
						// add p(y in the current interval)
						numRuns += nonOffsetProb;
					}
				} else {
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
			if (intervalStart != totalNumRows - 1) {
				intervalEnd = totalNumRows;
				intervalSize = intervalEnd - intervalStart - 1;
				// expected value of a multivariate hypergeometric distribution
				additionalOffsets = offsetsRatio * intervalSize;
				// expected value of an Ising-Stevens distribution
				numRuns += (intervalSize - additionalOffsets)
						* additionalOffsets / intervalSize;
				nonOffsetProb = (intervalSize - additionalOffsets)
						/ intervalSize;
			} else {
				nonOffsetProb = 1;
			}
			// additional runs resulting from x's on the boundaries of the
			// separators
			endedWithOffset = intervalStart == offsets[offsets.length - 1];
			if (seenNonOffset) {
				if (startedWithOffset) {
					numRuns += prevNonOffsetProb;
				}
				if (endedWithOffset) {
					// add p(y in the current interval)
					numRuns += nonOffsetProb;
				}
			} else {
				if (endedWithOffset)
					// add p(y in the previous interval and y in the current
					// interval)
					numRuns += prevNonOffsetProb * nonOffsetProb;
			}
		}
		return Math.round(numRuns);
	}

	/**
	 * 
	 * @param colIndexes
	 * @return
	 */
	private int haasAndStokes(int[] colIndexes) {
		ReaderColumnSelection reader =  new ReaderColumnSelectionDenseSample(_data, 
				colIndexes, _sampleRows, !CompressedMatrixBlock.MATERIALIZE_ZEROS);
		return haasAndStokes(_numRows, _sampleRows.length, reader);
	}

	/**
	 * TODO remove, just for local debugging.
	 * 
	 * @param colIndexes
	 * @return
	 */
	@SuppressWarnings("unused")
	private int getExactNumDistinctValues(int[] colIndexes) {
		HashSet<DblArray> distinctVals = new HashSet<DblArray>();
		ReaderColumnSelection reader = (_data.isInSparseFormat() && CompressedMatrixBlock.TRANSPOSE_INPUT) ? 
				new ReaderColumnSelectionSparse(_data, colIndexes, !CompressedMatrixBlock.MATERIALIZE_ZEROS) : 
				new ReaderColumnSelectionDense(_data, colIndexes, !CompressedMatrixBlock.MATERIALIZE_ZEROS);
		DblArray val = null;
		while (null != (val = reader.nextRow()))
			distinctVals.add(val);
		return distinctVals.size();
	}

	/**
	 * Returns a sorted array of n integers, drawn uniformly from the range [0,range).
	 * 
	 * @param range
	 * @param smplSize
	 * @return
	 */
	private int[] getSortedUniformSample(int range, int smplSize) {
		if (smplSize == 0)
			return new int[] {};
		int[] sample = _rng.nextPermutation(range, smplSize);
		Arrays.sort(sample);
		return sample;
	}
	

	/////////////////////////////////////////////////////
	// Sample Cardinality Estimator library
	/////////////////////////////////////////
	
	/**
	 * M. Charikar, S. Chaudhuri, R. Motwani, and V. R. Narasayya, Towards
	 * estimation error guarantees for distinct values, PODS'00.
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 *            : a reader for the sampled rows
	 * @return
	 */
	@SuppressWarnings("unused")
	private static int guaranteedErrorEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader) {
		HashMap<DblArray, Integer> valsCount = getValCounts(sampleRowsReader);
		// number of values that occur only once
		int singltonValsCount = 0;
		int otherValsCount = 0;
		for (Integer c : valsCount.values()) {
			if (c == 1)
				singltonValsCount++;
			else
				otherValsCount++;
		}
		return (int) Math.round(otherValsCount + singltonValsCount
				* Math.sqrt(((double) nRows) / sampleSize));
	}

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. 
	 * Sampling-Based Estimation of the Number of Distinct Values of an
	 * Attribute. VLDB'95, Section 3.2.
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @return
	 */
	@SuppressWarnings("unused")
	private static int shlosserEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader) 
	{
		return shlosserEstimator(nRows, sampleSize, sampleRowsReader,
				getValCounts(sampleRowsReader));
	}

	/**
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @param valsCount
	 * @return
	 */
	private static int shlosserEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader,
			HashMap<DblArray, Integer> valsCount) 
	{
		double q = ((double) sampleSize) / nRows;
		double oneMinusQ = 1 - q;

		int[] freqCounts = getFreqCounts(valsCount);

		double numerSum = 0, denomSum = 0;
		int iPlusOne = 1;
		for (int i = 0; i < freqCounts.length; i++, iPlusOne++) {
			numerSum += Math.pow(oneMinusQ, iPlusOne) * freqCounts[i];
			denomSum += iPlusOne * q * Math.pow(oneMinusQ, i) * freqCounts[i];
		}
		int estimate = (int) Math.round(valsCount.size() + freqCounts[0]
				* numerSum / denomSum);
		return estimate < 1 ? 1 : estimate;
	}

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes.
	 * Sampling-Based Estimation of the Number of Distinct Values of an
	 * Attribute. VLDB'95, Section 4.3.
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @return
	 */
	@SuppressWarnings("unused")
	private static int smoothedJackknifeEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader) 
	{
		return smoothedJackknifeEstimator(nRows, sampleSize, sampleRowsReader,
				getValCounts(sampleRowsReader));
	}

	/**
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @param valsCount
	 * @return
	 */
	private static int smoothedJackknifeEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader,
			HashMap<DblArray, Integer> valsCount) 
	{
		int[] freqCounts = getFreqCounts(valsCount);
		// all values in the sample are zeros
		if (freqCounts.length == 0)
			return 0;
		// nRows is N and sampleSize is n

		int d = valsCount.size();
		double f1 = freqCounts[0];
		int Nn = nRows * sampleSize;
		double D0 = (d - f1 / sampleSize)
				/ (1 - (nRows - sampleSize + 1) * f1 / Nn);
		double NTilde = nRows / D0;
		/*-
		 *
		 * h (as defined in eq. 5 in the paper) can be implemented as:
		 * 
		 * double h = Gamma(nRows - NTilde + 1) x Gamma.gamma(nRows -sampleSize + 1) 
		 * 		     ----------------------------------------------------------------
		 *  		Gamma.gamma(nRows - sampleSize - NTilde + 1) x Gamma.gamma(nRows + 1)
		 * 
		 * 
		 * However, for large values of nRows, Gamma.gamma returns NAN
		 * (factorial of a very large number).
		 * 
		 * The following implementation solves this problem by levaraging the
		 * cancelations that show up when expanding the factorials in the
		 * numerator and the denominator.
		 * 
		 * 
		 * 		min(A,D-1) x [min(A,D-1) -1] x .... x B
		 * h = -------------------------------------------
		 * 		C x [C-1] x .... x max(A+1,D)
		 * 
		 * where A = N-\tilde{N}
		 *       B = N-\tilde{N} - n + a
		 *       C = N
		 *       D = N-n+1
		 *       
		 * 		
		 *
		 */
		double A = (int) nRows - NTilde;
		double B = A - sampleSize + 1;
		double C = nRows;
		double D = nRows - sampleSize + 1;
		A = Math.min(A, D - 1);
		D = Math.max(A + 1, D);
		double h = 1;

		for (; A >= B || C >= D; A--, C--) {
			if (A >= B)
				h *= A;
			if (C >= D)
				h /= C;
		}
		// end of h computation

		double g = 0, gamma = 0;
		// k here corresponds to k+1 in the paper (the +1 comes from replacing n
		// with n-1)
		for (int k = 2; k <= sampleSize + 1; k++) {
			g += 1.0 / (nRows - NTilde - sampleSize + k);
		}
		for (int i = 1; i <= freqCounts.length; i++) {
			gamma += i * (i - 1) * freqCounts[i - 1];
		}
		gamma *= (nRows - 1) * D0 / Nn / (sampleSize - 1);
		gamma += D0 / nRows - 1;

		double estimate = (d + nRows * h * g * gamma)
				/ (1 - (nRows - NTilde - sampleSize + 1) * f1 / Nn);
		return estimate < 1 ? 1 : (int) Math.round(estimate);
	}

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. 1995.
	 * Sampling-Based Estimation of the Number of Distinct Values of an
	 * Attribute. VLDB'95, Section 5.2, recommended estimator by the authors
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @return
	 */
	@SuppressWarnings("unused")
	private static int shlosserJackknifeEstimator(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader) {
		HashMap<DblArray, Integer> valsCount = getValCounts(sampleRowsReader);

		// uniformity chi-square test
		double nBar = ((double) sampleSize) / valsCount.size();
		// test-statistic
		double u = 0;
		for (int cnt : valsCount.values()) {
			u += Math.pow(cnt - nBar, 2);
		}
		u /= nBar;
		if (sampleSize != usedSampleSize)
			computeCriticalValue(sampleSize);
		if (u < uniformityCriticalValue) {
			// uniform
			return smoothedJackknifeEstimator(nRows, sampleSize,
					sampleRowsReader, valsCount);
		} else {
			return shlosserEstimator(nRows, sampleSize, sampleRowsReader,
					valsCount);
		}
	}

	/*
	 * In the shlosserSmoothedJackknifeEstimator as long as the sample size did
	 * not change, we will have the same critical value each time the estimator
	 * is used (given that alpha is the same). We cache the critical value to
	 * avoid recomputing it in each call.
	 */
	private static double uniformityCriticalValue;
	private static int usedSampleSize;
	
	private static void computeCriticalValue(int sampleSize) {
		ChiSquaredDistribution chiSqr = new ChiSquaredDistribution(sampleSize - 1);
		uniformityCriticalValue = chiSqr.inverseCumulativeProbability(SHLOSSER_JACKKNIFE_ALPHA);
		usedSampleSize = sampleSize;
	}

	/**
	 * Haas, Peter J., and Lynne Stokes.
	 * "Estimating the number of classes in a finite population." Journal of the
	 * American Statistical Association 93.444 (1998): 1475-1487.
	 * 
	 * The hybrid estimator given by Eq. 33 in Section 6
	 * 
	 * @param nRows
	 * @param sampleSize
	 * @param sampleRowsReader
	 * @return
	 */
	private static int haasAndStokes(int nRows, int sampleSize,
			ReaderColumnSelection sampleRowsReader) 
	{
		HashMap<DblArray, Integer> valsCount = getValCounts(sampleRowsReader);
		// all values in the sample are zeros.
		if (valsCount.size() == 0)
			return 1;
		int[] freqCounts = getFreqCounts(valsCount);
		float q = ((float) sampleSize) / nRows;
		float _1MinusQ = 1 - q;
		// Eq. 11
		float duj1Fraction = ((float) sampleSize)
				/ (sampleSize - _1MinusQ * freqCounts[0]);
		float duj1 = duj1Fraction * valsCount.size();
		// Eq. 16
		float gamma = 0;
		for (int i = 1; i <= freqCounts.length; i++) {
			gamma += i * (i - 1) * freqCounts[i - 1];
		}
		gamma *= duj1 / sampleSize / sampleSize;
		gamma += duj1 / nRows - 1;
		gamma = Math.max(gamma, 0);
		int estimate;
		
		if (gamma < HAAS_AND_STOKES_ALPHA1) {
			// UJ2 - begining of page 1479
		//	System.out.println("uj2");
			estimate = (int) (duj1Fraction * (valsCount.size() - freqCounts[0]
					* _1MinusQ * Math.log(_1MinusQ) * gamma / q));
		} else if (gamma < HAAS_AND_STOKES_ALPHA2) {
			// UJ2a - end of page 1998
			//System.out.println("uj2a");
			int numRemovedClasses = 0;
			float updatedNumRows = nRows;
			int updatedSampleSize = sampleSize;

			for (Integer cnt : valsCount.values()) {
				if (cnt > HAAS_AND_STOKES_UJ2A_C) {
					numRemovedClasses++;
					freqCounts[cnt - 1]--;
					updatedSampleSize -= cnt;
					/*
					 * To avoid solving Eq. 20 numerically for the class size in
					 * the full population (N_j), the current implementation
					 * just scales cnt (n_j) by the sampling ratio (q).
					 * Intuitively, the scaling should be fine since cnt is
					 * large enough. Also, N_j in Eq. 20 is lower-bounded by cnt
					 * which is already large enough to make the denominator in
					 * Eq. 20 very close to 1.
					 */
					updatedNumRows -= ((float) cnt) / q;
				}
			}
			if (updatedSampleSize == 0) {
				// use uJ2a
				
				estimate = (int) (duj1Fraction * (valsCount.size() - freqCounts[0]
						* (_1MinusQ) * Math.log(_1MinusQ) * gamma / q));
			} else {
				float updatedQ = ((float) updatedSampleSize) / updatedNumRows;
				int updatedSampleCardinality = valsCount.size()
						- numRemovedClasses;
				float updatedDuj1Fraction = ((float) updatedSampleSize)
						/ (updatedSampleSize - (1 - updatedQ) * freqCounts[0]);
				float updatedDuj1 = updatedDuj1Fraction
						* updatedSampleCardinality;
				float updatedGamma = 0;
				for (int i = 1; i <= freqCounts.length; i++) {
					updatedGamma += i * (i - 1) * freqCounts[i - 1];
				}
				updatedGamma *= updatedDuj1 / updatedSampleSize
						/ updatedSampleSize;
				updatedGamma += updatedDuj1 / updatedNumRows - 1;
				updatedGamma = Math.max(updatedGamma, 0);

				estimate = (int) (updatedDuj1Fraction * (updatedSampleCardinality - freqCounts[0]
						* (1 - updatedQ)
						* Math.log(1 - updatedQ)
						* updatedGamma / updatedQ))
						+ numRemovedClasses;
			}

		} else {
			// Sh3 - end of section 3
			float fraq1Numer = 0;
			float fraq1Denom = 0;
			float fraq2Numer = 0;
			float fraq2Denom = 0;
			for (int i = 1; i <= freqCounts.length; i++) {
				fraq1Numer += i * q * q * Math.pow(1 - q * q, i - 1)
						* freqCounts[i - 1];
				fraq1Denom += Math.pow(_1MinusQ, i) * (Math.pow(1 + q, i) - 1)
						* freqCounts[i - 1];
				fraq2Numer += Math.pow(_1MinusQ, i) * freqCounts[i - 1];
				fraq2Denom += i * q * Math.pow(_1MinusQ, i - 1)
						* freqCounts[i - 1];
			}
			estimate = (int) (valsCount.size() + freqCounts[0] * fraq1Numer
					/ fraq1Denom * fraq2Numer * fraq2Numer / fraq2Denom
					/ fraq2Denom);
		}
		return estimate < 1 ? 1 : estimate;
	}

	/**
	 * 
	 * @param sampleRowsReader
	 * @return
	 */
	private static HashMap<DblArray, Integer> getValCounts(
			ReaderColumnSelection sampleRowsReader) 
	{
		HashMap<DblArray, Integer> valsCount = new HashMap<DblArray, Integer>();
		DblArray val = null;
		Integer cnt;
		while (null != (val = sampleRowsReader.nextRow())) {
			cnt = valsCount.get(val);
			if (cnt == null)
				cnt = 0;
			cnt++;
			valsCount.put(val, cnt);
		}
		return valsCount;
	}

	/**
	 * 
	 * @param valsCount
	 * @return
	 */
	private static int[] getFreqCounts(HashMap<DblArray, Integer> valsCount) 
	{
		int maxCount = 0;
		for (Integer c : valsCount.values()) {
			if (c > maxCount)
				maxCount = c;
		}
		
		/*
		 * freqCounts[i-1] = how many values occured with a frequecy i
		 */
		int[] freqCounts = new int[maxCount];
		for (Integer c : valsCount.values()) {
			freqCounts[c - 1]++;
		}
		return freqCounts;

	}
}
