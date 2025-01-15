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

package org.apache.sysds.runtime.compress.estim.sample;

import java.util.HashMap;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.UnivariateSolverUtils;

public interface HassAndStokes {
	// static final Log LOG = LogFactory.getLog(HassAndStokes.class.getName());

	public static final double HAAS_AND_STOKES_ALPHA1 = 0.9; // 0.9 recommended in paper
	public static final double HAAS_AND_STOKES_ALPHA2 = 30; // 30 recommended in paper;
	public static final int HAAS_AND_STOKES_UJ2A_C = 50; // 50 recommend in paper
	public static final boolean HAAS_AND_STOKES_UJ2A_CUT2 = true; // cut frequency in half
	public static final boolean HAAS_AND_STOKES_UJ2A_SOLVE = true; // true recommended
	public static final int MAX_SOLVE_CACHE_SIZE = 64 * 1024; // global 2MB cache

	/**
	 * Haas, Peter J., and Lynne Stokes. "Estimating the number of classes in a finite population." Journal of the
	 * American Statistical Association 93.444 (1998): 1475-1487.
	 * 
	 * The hybrid estimator given by Eq. 33 in Section 6
	 * 
	 * @param numVals    The number of unique values in the sample
	 * @param freqCounts The inverse histogram of frequencies. counts extracted
	 * @param nRows      The number of rows originally in the input
	 * @param sampleSize The number of rows used in the sample
	 * @param solveCache A Hashmap containing information for getDuj2aEstimate
	 * @return An estimation of distinct elements in the population.
	 */
	public static int distinctCount(int numVals, int[] freqCounts, int nRows, int sampleSize,
		HashMap<Integer, Double> solveCache) {

		double q = ((double) sampleSize) / nRows;
		double f1 = freqCounts[0];

		// compute basic Duj1 estimate
		double duj1 = getDuj1Estimate(q, f1, sampleSize, numVals);

		// compute gamma based on Duj1
		double gamma = getGammaSquared(duj1, freqCounts, sampleSize, nRows);
		double d = -1;

		// core hybrid estimator based on gamma
		if(gamma < HAAS_AND_STOKES_ALPHA1)
			d = getDuj2Estimate(q, f1, sampleSize, numVals, gamma);
		else if(gamma < HAAS_AND_STOKES_ALPHA2)
			d = getDuj2aEstimate(q, freqCounts, sampleSize, numVals, gamma, nRows, solveCache);
		else
			d = getSh3Estimate(q, freqCounts, numVals);

		// round and ensure min value 1
		return (int) Math.round(d);

	}

	private static double getDuj1Estimate(double q, double f1, int n, int dn) {
		// Computes the "un-smoothed first-order jackknife estimator" (Eq 11).
		return dn / (1 - ((1 - q) * f1) / n);
	}

	private static double getDuj2Estimate(double q, double f1, int n, int dn, double gammaDuj1) {
		// Computes the "un-smoothed second-order jackknife estimator" (Eq 18b).
		return (dn - (1 - q) * f1 * Math.log(1 - q) * gammaDuj1 / q) / (1 - ((1 - q) * f1) / n);
	}

	private static double getDuj2aEstimate(double q, int f[], int n, int dn, double gammaDuj1, int N,
		HashMap<Integer, Double> solveCache) {
		// Computes the "un-smoothed second-order jackknife estimator" with additional stabilization procedure, which
		// removes the classes whose frequency exceed c, computes Duj2 over the reduced sample, and finally adds the
		// removed frequencies.
		int c = HAAS_AND_STOKES_UJ2A_CUT2 ? f.length / 2 + 1 : HAAS_AND_STOKES_UJ2A_C + 1;

		// compute adjusted sample size after removing classes that
		// exceed a fixed frequency c
		int nB = 0, cardB = 0;
		for(int i = c; i <= f.length; i++)
			if(f[i - 1] != 0) {
				nB += f[i - 1] * i; // numVals times frequency
				cardB += f[i - 1];
			}

		// fallback to Duj2 over full sample if only high frequency columns
		// This fallback is never hit therefore commented out.
		// if(n - nB == 0)
			// return getDuj2Estimate(q, f[0], n, dn, gammaDuj1);

		// compute reduced population size via numeric solve
		int updatedN = N;
		if(solveCache == null) {
			for(int i = c; i <= f.length; i++)
				if(f[i - 1] != 0)
					updatedN -= f[i - 1] * (!HAAS_AND_STOKES_UJ2A_SOLVE ? i / q : getMethodOfMomentsEstimate(i, q, 1, N));
		}
		else {
			for(int i = c; i <= f.length; i++)
				if(f[i - 1] != 0)
					updatedN -= f[i - 1] *
						(!HAAS_AND_STOKES_UJ2A_SOLVE ? i / q : getMethodOfMomentsEstimateWithCache(i, q, 1, N, solveCache));
		}

		// remove classes that exceed a fixed frequency c
		for(int i = c; i <= f.length; i++)
			f[i - 1] = 0;

		// compute duj2a over reduced sample
		double updatedDuj1 = getDuj1Estimate(q, f[0], n - nB, dn - cardB);
		double updatedGammaDuj1 = getGammaSquared(updatedDuj1, f, n - nB, updatedN);
		double duj2 = getDuj2Estimate(q, f[0], n - nB, dn - cardB, updatedGammaDuj1);
		return duj2 + cardB;
	}

	private static double getGammaSquared(double D, int[] f, int n, int N) {
		// Computes the "squared coefficient of variation" based on a given initial estimate D (Eq 16).
		double gamma = 0;
		for(int i = 2; i <= f.length; i++){
			int im1 = i - 1;
			// if(f[im1] != 0)
				gamma += i * (im1) * f[im1];
		}
		gamma *= D / n / n;
		gamma += D / N - 1;
		return Math.max(0, gamma);
	}

	private static double getSh3Estimate(double q, int[] f, double dn) {
		// Computed the "shlosser third-order estimator". (Eq 30b)

		// Note that this estimator can show anomalies with NaN as the results due to terms such as Math.pow(1+q, i) which
		// exceed Double.MAX_VALUE even for moderately large i, e.g., q=0.05 at around 14K.

		double fraq11 = 0, fraq12 = 0, fraq21 = 0, fraq22 = 0;
		for(int i = 1; i <= f.length; i++)
			if(f[i - 1] != 0) {
				fraq11 += i * q * q * Math.pow(1 - q * q, i - 1) * f[i - 1];
				// NOTE: numerically unstable due to Math.pow(1+q, i) overflows
				// fraq12 += Math.pow(1 - q, i) * (Math.pow(1+q, i)-1) * f[i-1];
				fraq12 += (Math.pow(1 - q * q, i) - Math.pow(1 - q, i)) * f[i - 1];
				fraq21 += Math.pow(1 - q, i) * f[i - 1];
				fraq22 += i * q * Math.pow(1 - q, i - 1) * f[i - 1];
			}
		return dn + f[0] * fraq11 / fraq12 * Math.pow(fraq21 / fraq22, 2);
	}

	private static double getMethodOfMomentsEstimateWithCache(int nj, double q, double min, double max,
		HashMap<Integer, Double> solveCache) {

		if(solveCache.containsKey(nj))
			synchronized(solveCache) {
				return solveCache.get(nj);
			}

		double est = getMethodOfMomentsEstimate(nj, q, min, max);

		if(solveCache.size() < MAX_SOLVE_CACHE_SIZE)
			synchronized(solveCache) {
				solveCache.put(nj, est);
			}

		return est;
	}

	private static double getMethodOfMomentsEstimate(int nj, double q, double min, double max) {
		// Solves the method-of-moments estimate numerically. We use a cache on the same observed instances in the sample
		// as q is constant and min/max are chosen conservatively.
		return UnivariateSolverUtils.solve(new MethodOfMomentsFunction(nj, q), min, max, 1e-9);
	}

	public static class MethodOfMomentsFunction implements UnivariateFunction {
		private final int _nj;
		private final double _q;

		public MethodOfMomentsFunction(int nj, double q) {
			_nj = nj;
			_q = q;
		}

		@Override
		public double value(double x) {
			return _q * x / (1 - Math.pow(1 - _q, x)) - _nj;
		}
	}
}
