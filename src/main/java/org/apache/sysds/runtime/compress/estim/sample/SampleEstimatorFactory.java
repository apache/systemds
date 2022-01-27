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

import java.util.Arrays;
import java.util.HashMap;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public class SampleEstimatorFactory {

	// private static final Log LOG = LogFactory.getLog(SampleEstimatorFactory.class.getName());

	public enum EstimationType {
		HassAndStokes, ShlosserEstimator, ShlosserJackknifeEstimator, SmoothedJackknifeEstimator,
	}

	/**
	 * estimate a distinct count based on frequencies
	 * 
	 * @param frequencies A list of frequencies of unique values
	 * @param nRows       The total number of rows to consider
	 * @param sampleSize  The size of the sample, NOTE this should ideally be scaled to match the sum(frequencies)
	 * @param type        the Type of estimator to use
	 * @param solveCache  A solve cache to avoid repeated calculations
	 * @return A bounded number of unique values.
	 */
	public static int distinctCount(int[] frequencies, int nRows, int sampleSize, EstimationType type,
		HashMap<Integer, Double> solveCache) {

		if(frequencies == null || frequencies.length == 0)
			// Frequencies for some reason is allocated as null or all values in the sample are zeros.
			return 0;

		try {
			// Invert histogram
			int[] invHist = getInvertedFrequencyHistogram(frequencies);
			// estimate distinct
			int est = distinctCountWithHistogram(frequencies.length, invHist, frequencies, nRows, sampleSize, type,
				solveCache);
			// Number of unique is trivially bounded by 
			// lower: the number of observed uniques in the sample
			// upper: the number of rows minus the observed uniques total count, plus the observed number of uniques.
			return Math.min(Math.max(frequencies.length, est), nRows - sampleSize + frequencies.length);
		}
		catch(Exception e) {
			throw new DMLCompressionException(
				"Error while estimating distinct count with arguments:\n\tfrequencies:" + Arrays.toString(frequencies)
					+ " nrows: " + nRows + " sampleSize: " + sampleSize + " type: " + type + " solveCache: " + solveCache,
				e);
		}
	}

	public static int distinctCountWithHistogram(int numVals, int[] invHist, int[] frequencies, int nRows,
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
		for(int i = 0; i < numVals; i++) {
			if(frequencies[i] != 0)
				freqCounts[frequencies[i] - 1]++;
		}

		return freqCounts;
	}
}
