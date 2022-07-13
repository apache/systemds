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

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

public interface ShlosserJackknifeEstimator {

	final static double SHLOSSER_JACKKNIFE_ALPHA = 0.975;

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. 1995. Sampling-Based Estimation of the Number
	 * of Distinct Values of an Attribute. VLDB'95, Section 5.2, recommended estimator by the authors
	 * 
	 * @param numVals	 The number of unique values in the sample
	 * @param frequencies The Frequencies of the different unique values
	 * @param freqCounts The inverse histogram of frequencies. counts extracted 
	 * @param nRows      The original number of rows in the entire input
	 * @param sampleSize The number of rows in the sample
	 * @return an estimation of number of distinct values.
	 */
	public static int distinctCount(int numVals, int[] frequencies, int[] freqCounts, int nRows, int sampleSize) {

		final CriticalValue cv = computeCriticalValue(sampleSize);

		// uniformity chi-square test
		double nBar = ((double) sampleSize) / numVals;
		// test-statistic
		double u = 0;
		for(int i = 0; i < numVals; i++)
			u += Math.pow(frequencies[i] - nBar, 2);
		
		u /= nBar;

		if(u < cv.uniformityCriticalValue) // uniform
			return SmoothedJackknifeEstimator.distinctCount(numVals, freqCounts, nRows, sampleSize);
		else
			return ShlosserEstimator.distinctCount(numVals, freqCounts, nRows, sampleSize);
	}

	private static CriticalValue computeCriticalValue(int sampleSize) {
		ChiSquaredDistribution chiSqr = new ChiSquaredDistribution(sampleSize - 1);
		return new CriticalValue(chiSqr.inverseCumulativeProbability(SHLOSSER_JACKKNIFE_ALPHA), sampleSize);
	}

	/*
	 * In the shlosserSmoothedJackknifeEstimator as long as the sample size did not change, we will have the same
	 * critical value each time the estimator is used (given that alpha is the same). We cache the critical value to
	 * avoid recomputing it in each call.
	 */
	public static class CriticalValue {
		public final double uniformityCriticalValue;
		public final int usedSampleSize;

		public CriticalValue(double cv, int size) {
			uniformityCriticalValue = cv;
			usedSampleSize = size;
		}
	}
}
