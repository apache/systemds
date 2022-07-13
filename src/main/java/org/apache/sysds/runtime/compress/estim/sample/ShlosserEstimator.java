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

public interface ShlosserEstimator {

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. Sampling-Based Estimation of the Number of
	 * Distinct Values of an Attribute. VLDB'95, Section 3.2.
	 * 
	 * @param numVals    The number of unique values in the sample
	 * @param freqCounts The inverse histogram of frequencies. counts extracted
	 * @param nRows      The original number of rows in the entire input
	 * @param sampleSize The number of rows in the sample
	 * @return an estimation of number of distinct values.
	 */
	public static int distinctCount(long numVals, int[] freqCounts, long nRows, long sampleSize) {

		if(freqCounts[0] == 0) // early abort
			return (int) numVals;

		final double q = ((double) sampleSize) / nRows;
		final double oneMinusQ = 1 - q;

		double numberSum = 0, denomSum = 0, p1 = 0;

		int i = 0;
		while(i < freqCounts.length) {
			p1 = Math.pow(oneMinusQ, i) * freqCounts[i];
			numberSum += p1 * oneMinusQ;
			denomSum += (++i) * q * p1;
		}

		if(denomSum == 0 || denomSum == Double.POSITIVE_INFINITY || denomSum == Double.NaN)
			return (int) numVals;

		return (int) Math.round(numVals + freqCounts[0] * numberSum / denomSum);

	}
}
