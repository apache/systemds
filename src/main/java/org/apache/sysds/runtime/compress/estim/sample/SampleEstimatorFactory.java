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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public interface SampleEstimatorFactory {

	static final Log LOG = LogFactory.getLog(SampleEstimatorFactory.class.getName());

	public enum EstimationType {
		HassAndStokes, ShlosserEstimator, //
		ShlosserJackknifeEstimator, SmoothedJackknifeEstimator
	}

	/**
	 * Estimate a distinct number of values based on frequencies.
	 * 
	 * @param frequencies A list of frequencies of unique values, NOTE all values contained should be larger than zero
	 * @param nRows       The total number of rows to consider, NOTE should always be larger or equal to sum(frequencies)
	 * @param sampleSize  The size of the sample, NOTE this should ideally be scaled to match the sum(frequencies) and
	 *                    should always be lower or equal to nRows
	 * @param type        The type of estimator to use
	 * @return A estimated number of unique values
	 */
	public static int distinctCount(int[] frequencies, int nRows, int sampleSize, EstimationType type) {
		return distinctCount(frequencies, nRows, sampleSize, type, null);
	}

	/**
	 * Estimate a distinct number of values based on frequencies.
	 * 
	 * @param frequencies A list of frequencies of unique values, NOTE all values contained should be larger than zero!
	 * @param nRows       The total number of rows to consider, NOTE should always be larger or equal to sum(frequencies)
	 * @param sampleSize  The size of the sample, NOTE this should ideally be scaled to match the sum(frequencies) and
	 *                    should always be lower or equal to nRows
	 * @param type        The type of estimator to use
	 * @param solveCache  A solve cache to avoid repeated calculations
	 * @return A estimated number of unique values
	 */
	public static int distinctCount(int[] frequencies, int nRows, int sampleSize, EstimationType type,
		HashMap<Integer, Double> solveCache) {
		if(frequencies == null || frequencies.length == 0)
			// Frequencies for some reason is allocated as null or all values in the sample are zeros.
			return 0;

		// Invert histogram
		final int[] invHist = getInvertedFrequencyHistogram(frequencies);
		// estimate distinct
		final int est = distinctCountWithHistogram(frequencies.length, invHist, frequencies, nRows, sampleSize, type,
			solveCache);
		// Number of unique is trivially bounded by:
		// lower: The number of observed uniques in the sample
		final int low = Math.max(frequencies.length, est);
		// upper: The number of rows minus the observed uniques total count, plus the observed number of uniques.
		final int high = Math.min(low, nRows - sampleSize + frequencies.length);
		return high;
	}

	private static int distinctCountWithHistogram(int numVals, int[] invHist, int[] frequencies, int nRows,
		int sampleSize, EstimationType type, HashMap<Integer, Double> solveCache) {
		switch(type) {
			case ShlosserEstimator:
				return ShlosserEstimator.distinctCount(numVals, invHist, nRows, sampleSize);
			case ShlosserJackknifeEstimator:
				return ShlosserJackknifeEstimator.distinctCount(numVals, frequencies, invHist, nRows, sampleSize);
			case SmoothedJackknifeEstimator:
				return SmoothedJackknifeEstimator.distinctCount(numVals, invHist, nRows, sampleSize);
			case HassAndStokes:
			default:
				return HassAndStokes.distinctCount(numVals, invHist, nRows, sampleSize, solveCache);
		}
	}

	private static int[] getInvertedFrequencyHistogram(int[] frequencies) {
		final int numVals = frequencies.length;
		// Find max
		int maxCount = 0;
		for(int i = 0; i < numVals; i++) {
			final int v = frequencies[i];
			if(v > maxCount)
				maxCount = v;
		}

		// create frequency histogram
		int[] freqCounts = new int[maxCount];
		for(int i = 0; i < numVals; i++)
			freqCounts[frequencies[i] - 1]++;

		return freqCounts;
	}
}
