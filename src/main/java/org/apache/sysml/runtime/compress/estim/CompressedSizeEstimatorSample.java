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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.UnivariateSolverUtils;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.BitmapEncoder;
import org.apache.sysml.runtime.compress.ReaderColumnSelection;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.compress.UncompressedBitmap;
import org.apache.sysml.runtime.compress.utils.DblArray;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorSample extends CompressedSizeEstimator 
{
	private final static double SHLOSSER_JACKKNIFE_ALPHA = 0.975;
	public static final double HAAS_AND_STOKES_ALPHA1 = 0.9; //0.9 recommended in paper
	public static final double HAAS_AND_STOKES_ALPHA2 = 30; //30 recommended in paper
	public static final int HAAS_AND_STOKES_UJ2A_C = 50; //50 recommend in paper
	public static final boolean HAAS_AND_STOKES_UJ2A_CUT2 = true; //cut frequency in half
	public static final boolean HAAS_AND_STOKES_UJ2A_SOLVE = true; //true recommended
	public static final int MAX_SOLVE_CACHE_SIZE = 64*1024; //global 2MB cache
	
	private static final Log LOG = LogFactory.getLog(CompressedSizeEstimatorSample.class.getName());
    
    private int[] _sampleRows = null;
    private HashMap<Integer, Double> _solveCache = null;
	
	public CompressedSizeEstimatorSample(MatrixBlock data, int sampleSize) 
		throws DMLRuntimeException 
	{
		super(data);
		
		//get sample of rows, incl eager extraction 
		_sampleRows = getSortedUniformSample(_numRows, sampleSize);
		if( SizeEstimatorFactory.EXTRACT_SAMPLE_ONCE ) {
			MatrixBlock select = new MatrixBlock(_numRows, 1, false);
			for( int i=0; i<sampleSize; i++ )
				select.quickSetValue(_sampleRows[i], 0, 1);
			_data = _data.removeEmptyOperations(new MatrixBlock(), 
					!CompressedMatrixBlock.TRANSPOSE_INPUT, select);
		}
		
		//establish estimator-local cache for numeric solve
		_solveCache = new HashMap<Integer, Double>();
	}

	@Override
	public CompressedSizeInfo estimateCompressedColGroupSize(int[] colIndexes) 
	{
		int sampleSize = _sampleRows.length;
		int numCols = colIndexes.length;
		int[] sampleRows = _sampleRows;
		
		//extract statistics from sample
		UncompressedBitmap ubm = SizeEstimatorFactory.EXTRACT_SAMPLE_ONCE ?
				BitmapEncoder.extractBitmap(colIndexes, _data) :
				BitmapEncoder.extractBitmapFromSample(colIndexes, _data, sampleRows);
		SizeEstimationFactors fact = computeSizeEstimationFactors(ubm, false);
		
		//estimate number of distinct values (incl fixes for anomalies w/ large sample fraction)
		int totalCardinality = getNumDistinctValues(ubm, _numRows, sampleRows, _solveCache);
		totalCardinality = Math.max(totalCardinality, fact.numVals);
		totalCardinality = Math.min(totalCardinality, _numRows); 
		
		//estimate unseen values
		int unseenVals = totalCardinality - fact.numVals;
		
		//estimate number of non-zeros (conservatively round up)
		double C = Math.max(1 - (double)fact.numSingle/sampleSize, (double)sampleSize/_numRows); 
		int numZeros = sampleSize - fact.numOffs; //>=0
		int numNonZeros = (int)Math.ceil(_numRows - (double)_numRows/sampleSize * C * numZeros);
		numNonZeros = Math.max(numNonZeros, totalCardinality); //handle anomaly of zi=0

		if( totalCardinality<=0 || unseenVals<0 || numZeros<0 || numNonZeros<=0 )
			LOG.warn("Invalid estimates detected for "+Arrays.toString(colIndexes)+": "
					+totalCardinality+" "+unseenVals+" "+numZeros+" "+numNonZeros);
			
		// estimate number of segments and number of runs incl correction for
		// empty segments and empty runs (via expected mean of offset value)
		int numUnseenSeg = (int) (unseenVals * 
			Math.ceil((double)_numRows/BitmapEncoder.BITMAP_BLOCK_SZ/2));
		int totalNumSeg = fact.numSegs + numUnseenSeg;
		int totalNumRuns = getNumRuns(ubm, sampleSize, _numRows, sampleRows) + numUnseenSeg;

		//construct new size info summary
		return new CompressedSizeInfo(totalCardinality, numNonZeros,
				getRLESize(totalCardinality, totalNumRuns, numCols),
				getOLESize(totalCardinality, numNonZeros, totalNumSeg, numCols),
				getDDCSize(totalCardinality, _numRows, numCols));
	}

	@Override
	public CompressedSizeInfo estimateCompressedColGroupSize(UncompressedBitmap ubm) 
	{
		//compute size estimation factors
		SizeEstimationFactors fact = computeSizeEstimationFactors(ubm, true);
		
		//construct new size info summary
		return new CompressedSizeInfo(fact.numVals, fact.numOffs,
				getRLESize(fact.numVals, fact.numRuns, ubm.getNumColumns()),
				getOLESize(fact.numVals, fact.numOffs, fact.numSegs, ubm.getNumColumns()),
				getDDCSize(fact.numVals, _numRows, ubm.getNumColumns()));
	}

	private static int getNumDistinctValues(UncompressedBitmap ubm, int numRows, int[] sampleRows, 
			HashMap<Integer, Double> solveCache) {
		return haasAndStokes(ubm, numRows, sampleRows.length, solveCache);
	}

	private static int getNumRuns(UncompressedBitmap ubm,
			int sampleSize, int totalNumRows, int[] sampleRows) {
		int numVals = ubm.getNumValues();
		// all values in the sample are zeros
		if (numVals == 0)
			return 0;
		double numRuns = 0;
		for (int vi = 0; vi < numVals; vi++) {
			int[] offsets = ubm.getOffsetsList(vi).extractValues();
			int offsetsSize = ubm.getNumOffsets(vi);
			double offsetsRatio = ((double) offsetsSize) / sampleSize;
			double avgAdditionalOffsets = offsetsRatio * totalNumRows
					/ sampleSize;
			if (avgAdditionalOffsets < 1) {
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
			if (sampleRows[0] == 0) {
				// empty interval
				intervalStart = 0;
			} else {
				intervalEnd = sampleRows[0];
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

				if (offsetsPtrs < offsetsSize
						&& offsets[offsetsPtrs] == intervalStart) {
					startedWithOffset = true;
					offsetsPtrs++;
					endedWithOffset = true;
				} else {
					seenNonOffset = true;
					endedWithOffset = false;
				}
				while (intervalStart + 1 == sampleRows[ix]) {
					intervalStart = sampleRows[ix];
					if (seenNonOffset) {
						if (offsetsPtrs < offsetsSize
								&& offsets[offsetsPtrs] == intervalStart) {
							withinSepRun = 1;
							offsetsPtrs++;
							endedWithOffset = true;
						} else {
							numRuns += withinSepRun;
							withinSepRun = 0;
							endedWithOffset = false;
						}
					} else if (offsetsPtrs < offsetsSize
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
				intervalEnd = sampleRows[ix];
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
			endedWithOffset = intervalStart == offsets[offsetsSize - 1];
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
		return (int)Math.min(Math.round(numRuns), Integer.MAX_VALUE);
	}

	/**
	 * Returns a sorted array of n integers, drawn uniformly from the range [0,range).
	 * 
	 * @param range the range
	 * @param smplSize sample size
	 * @return sorted array of integers
	 */
	private static int[] getSortedUniformSample(int range, int smplSize) {
		if (smplSize == 0)
			return new int[] {};
		RandomDataGenerator rng = new RandomDataGenerator();
		int[] sample = rng.nextPermutation(range, smplSize);
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
	 * @param nRows number of rows
	 * @param sampleSize sample size
	 * @param sampleRowsReader
	 *             a reader for the sampled rows
	 * @return error estimator
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
	 * @param nRows number of rows
	 * @param sampleSize sample size
	 * @param sampleRowsReader reader
	 * @return estimator
	 */
	private static int shlosserEstimator(UncompressedBitmap ubm, int nRows, int sampleSize) 
	{
		double q = ((double) sampleSize) / nRows;
		double oneMinusQ = 1 - q;

		int numVals = ubm.getNumValues();
		int[] freqCounts = getFreqCounts(ubm);

		double numerSum = 0, denomSum = 0;
		int iPlusOne = 1;
		for (int i = 0; i < freqCounts.length; i++, iPlusOne++) {
			numerSum += Math.pow(oneMinusQ, iPlusOne) * freqCounts[i];
			denomSum += iPlusOne * q * Math.pow(oneMinusQ, i) * freqCounts[i];
		}
		int estimate = (int) Math.round(numVals + freqCounts[0]
				* numerSum / denomSum);
		return estimate < 1 ? 1 : estimate;
	}

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes.
	 * Sampling-Based Estimation of the Number of Distinct Values of an
	 * Attribute. VLDB'95, Section 4.3.
	 * 
	 * @param nRows number of rows
	 * @param sampleSize sample size
	 * @param sampleRowsReader row reader
	 * @return estimator
	 */
	private static int smoothedJackknifeEstimator(UncompressedBitmap ubm, int nRows, int sampleSize) 
	{
		int numVals = ubm.getNumValues();
		int[] freqCounts = getFreqCounts(ubm);
		// all values in the sample are zeros
		if (freqCounts.length == 0)
			return 0;
		// nRows is N and sampleSize is n

		int d = numVals;
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
	 * @param nRows number of rows
	 * @param sampleSize sample size
	 * @param sampleRowsReader row reader
	 * @return estimator
	 */
	@SuppressWarnings("unused")
	private static int shlosserJackknifeEstimator(UncompressedBitmap ubm, int nRows, int sampleSize) 
	{
		int numVals = ubm.getNumValues();
		CriticalValue cv = computeCriticalValue(sampleSize);
		
		// uniformity chi-square test
		double nBar = ((double) sampleSize) / numVals;
		// test-statistic
		double u = 0;
		for( int i=0; i<numVals; i++ ) {
			u += Math.pow(ubm.getNumOffsets(i) - nBar, 2);
		}
		u /= nBar;
		if (sampleSize != cv.usedSampleSize)
			computeCriticalValue(sampleSize);
		if (u < cv.uniformityCriticalValue) // uniform
			return smoothedJackknifeEstimator(ubm, nRows, sampleSize);
		else 
			return shlosserEstimator(ubm, nRows, sampleSize);
	}

	private static CriticalValue computeCriticalValue(int sampleSize) {
		ChiSquaredDistribution chiSqr = new ChiSquaredDistribution(sampleSize - 1);
		return new CriticalValue(
			chiSqr.inverseCumulativeProbability(SHLOSSER_JACKKNIFE_ALPHA), sampleSize);
	}

	/**
	 * Haas, Peter J., and Lynne Stokes.
	 * "Estimating the number of classes in a finite population." Journal of the
	 * American Statistical Association 93.444 (1998): 1475-1487.
	 * 
	 * The hybrid estimator given by Eq. 33 in Section 6
	 * 
	 * @param nRows number of rows
	 * @param sampleSize sample size
	 * @param solveCache 
	 * @param sampleRowsReader row reader
	 * @return estimator
	 */
	private static int haasAndStokes(UncompressedBitmap ubm, int nRows, int sampleSize, HashMap<Integer, Double> solveCache)
	{
		//obtain value and frequency histograms
		int numVals = ubm.getNumValues();
		int[] freqCounts = getFreqCounts(ubm);
	
		// all values in the sample are zeros.
		if( numVals == 0 )
			return 1;
		
		double q = ((double) sampleSize) / nRows;
		double f1 = freqCounts[0];
		
		//compute basic Duj1 estimate
		double duj1 = getDuj1Estimate(q, f1, sampleSize, numVals);
		
		//compute gamma based on Duj1
		double gamma = getGammaSquared(duj1, freqCounts, sampleSize, nRows);
		double d = -1;
		
		//core hybrid estimator based on gamma
		if (gamma < HAAS_AND_STOKES_ALPHA1)
			d = getDuj2Estimate(q, f1, sampleSize, numVals, gamma);
		else if (gamma < HAAS_AND_STOKES_ALPHA2)
			d = getDuj2aEstimate(q, freqCounts, sampleSize, numVals, gamma, nRows, solveCache);
		else
			d = getSh3Estimate(q, freqCounts, numVals);
		
		//round and ensure min value 1
		return Math.max(1, (int)Math.round(d));
	}

	private static HashMap<DblArray, Integer> getValCounts(ReaderColumnSelection sampleRowsReader) 
	{
		HashMap<DblArray, Integer> valsCount = new HashMap<DblArray, Integer>();
		DblArray val = null;
		Integer cnt;
		while (null != (val = sampleRowsReader.nextRow())) {
			cnt = valsCount.get(val);
			if (cnt == null)
				cnt = 0;
			cnt++;
			valsCount.put(new DblArray(val), cnt);
		}
		return valsCount;
	}

	/**
	 * Creates an inverted histogram, where freqCounts[i-1] indicates 
	 * how many values occurred with a frequency i. Note that freqCounts[0]
	 * represents the special values of the number of singletons. 
	 * 
	 * @param ubm uncompressed bitmap
	 * @return frequency counts
	 */
	private static int[] getFreqCounts(UncompressedBitmap ubm) 
	{
		//determine max frequency
		int numVals = ubm.getNumValues();
		int maxCount = 0;
		for( int i=0; i<numVals; i++ )
			maxCount = Math.max(maxCount, ubm.getNumOffsets(i));
			
		//create frequency histogram
		int[] freqCounts = new int[maxCount];
		for( int i=0; i<numVals; i++ )
			freqCounts[ubm.getNumOffsets(i)-1] ++;

		return freqCounts;

	}

	/**
	 * Computes the "unsmoothed first-order jackknife estimator" (Eq 11).
	 * 
	 */
	private static double getDuj1Estimate(double q, double f1, int n, int dn) {
		return dn / (1 - ((1-q) * f1)/n);
	}
	
	/**
	 * Computes the "unsmoothed second-order jackknife estimator" (Eq 18b).
	 * 
	 */
	private static double getDuj2Estimate(double q, double f1, int n, int dn, double gammaDuj1) {
		return (dn - (1-q) * f1 * Math.log(1-q) * gammaDuj1 / q) / (1 - ((1-q) * f1)/n);
	}
	
	/**
	 * Computes the "unsmoothed second-order jackknife estimator" with additional  
	 * stabilization procedure, which removes the classes whose frequency exceed c,
	 * computes Duj2 over the reduced sample, and finally adds the removed frequencies.
	 * 
	 */
	private static double getDuj2aEstimate(double q, int f[], int n, int dn, double gammaDuj1, int N, 
			HashMap<Integer, Double> solveCache) {
		int c = HAAS_AND_STOKES_UJ2A_CUT2 ? 
			f.length/2+1 : HAAS_AND_STOKES_UJ2A_C+1;
		
		//compute adjusted sample size after removing classes that
		//exceed a fixed frequency  c
		int nB = 0, cardB = 0;
		for( int i=c; i<=f.length; i++ ) 
			if( f[i-1] != 0 ) {
				nB += f[i-1] * i; //numVals times frequency 
				cardB += f[i-1];
			}
		
		//fallback to Duj2 over full sample if only high frequency columns
		if( n - nB == 0 )
			return getDuj2Estimate(q, f[0], n, dn, gammaDuj1);

		//compute reduced population size via numeric solve
		int updatedN = N; 
		for( int i=c; i<=f.length; i++ )
			if( f[i-1] != 0 )
				updatedN -= f[i-1] * (!HAAS_AND_STOKES_UJ2A_SOLVE ? i/q :
					getMethodOfMomentsEstimate(i, q, 1, N, solveCache));
		
		//remove classes that exceed a fixed frequency c
		for( int i=c; i<=f.length; i++ )
			f[i-1] = 0; 
		
		//compute duj2a over reduced sample
		double updatedDuj1 = getDuj1Estimate(q, f[0], n-nB, dn-cardB);
		double updatedGammaDuj1 = getGammaSquared(updatedDuj1, f, n-nB, updatedN);
		double duj2 = getDuj2Estimate(q, f[0], n-nB, dn-cardB, updatedGammaDuj1);
		return duj2 + cardB;		
	}
	
	/**
	 * Computed the "shlosser third-order estimator". (Eq 30b)
	 * 
	 * Note that this estimator can show anomalies with NaN as the results
	 * due to terms such as Math.pow(1+q, i) which exceed Double.MAX_VALUE
	 * even for moderately large i, e.g., q=0.05 at around 14K.
	 * 
	 */
	private static double getSh3Estimate(double q, int[] f, double dn) {
		double fraq11 = 0, fraq12 = 0, fraq21 = 0, fraq22 = 0;
		for( int i=1; i<=f.length; i++ ) 
			if( f[i-1] != 0 ) {
				fraq11 += i * q*q * Math.pow(1 - q*q, i-1) * f[i-1];
				//NOTE: numerically unstable due to Math.pow(1+q, i) overflows
				//fraq12 += Math.pow(1 - q, i) * (Math.pow(1+q, i)-1) * f[i-1];
				fraq12 += (Math.pow(1 - q*q, i) - Math.pow(1 - q, i)) * f[i-1];
				fraq21 += Math.pow(1 - q, i) * f[i-1];
				fraq22 += i * q * Math.pow(1 - q, i-1) * f[i-1];
			}
		return dn + f[0] * fraq11/fraq12 * Math.pow(fraq21/fraq22, 2); 
	}
	
	/**
	 * Computes the "squared coefficient of variation" based on a given 
	 * initial estimate D (Eq 16).
	 * 
	 */
	private static double getGammaSquared(double D, int[] f, int n, int N) {
		double gamma = 0;
		for( int i=1; i<=f.length; i++) 
			if( f[i-1] != 0 )
				gamma += i * (i-1) * f[i-1];
		gamma *= D / n / n;
		gamma += D / N - 1;
		return Math.max(0, gamma);
	}
	
	/**
	 * Solves the method-of-moments estimate numerically. We use a cache
	 * on the same observed instances in the sample as q is constant and
	 * min/max are chosen conservatively.
	 * 
	 */
	private static double getMethodOfMomentsEstimate(int nj, double q, double min, double max, 
		HashMap<Integer, Double> solveCache) {
		if( solveCache.containsKey(nj) )
			return solveCache.get(nj);
		
		double est = UnivariateSolverUtils
			.solve(new MethodOfMomentsFunction(nj, q), min, max, 1e-9);
		
		if( solveCache.size()<MAX_SOLVE_CACHE_SIZE )
			solveCache.put(nj, est);
		
		return est;
	}
	
	/*
	 * In the shlosserSmoothedJackknifeEstimator as long as the sample size did
	 * not change, we will have the same critical value each time the estimator
	 * is used (given that alpha is the same). We cache the critical value to
	 * avoid recomputing it in each call.
	 */
	private static class CriticalValue {
		public final double uniformityCriticalValue;
		public final int usedSampleSize;
		
		public CriticalValue(double cv, int size) {
			uniformityCriticalValue = cv;
			usedSampleSize = size;
		} 
	}
	
	private static class MethodOfMomentsFunction implements UnivariateFunction {
		private final int _nj;
		private final double _q;
		
		public MethodOfMomentsFunction(int nj, double q) {
			_nj = nj;
			_q = q;
		}
		
		@Override
		public double value(double x) {
			return _q*x / (1-Math.pow(1-_q, x)) - _nj;
		}
	}
}
