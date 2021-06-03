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

public class ShlosserEstimator {

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. Sampling-Based Estimation of the Number of
	 * Distinct Values of an Attribute. VLDB'95, Section 3.2.
	 * 
	 * @param numVals	 The number of unique values in the sample
	 * @param freqCounts The inverse histogram of frequencies. counts extracted 
	 * @param nRows      The original number of rows in the entire input
	 * @param sampleSize The number of rows in the sample
	 * @return an estimation of number of distinct values.
	 */
	protected static int distinctCount(int numVals, int[] freqCounts, int nRows, int sampleSize) {
		double q = ((double) sampleSize) / nRows;
		double oneMinusQ = 1 - q;

		double numerSum = 0, denomSum = 0;
		int iPlusOne = 1;
		for(int i = 0; i < freqCounts.length; i++, iPlusOne++) {
			numerSum += Math.pow(oneMinusQ, iPlusOne) * freqCounts[i];
			denomSum += iPlusOne * q * Math.pow(oneMinusQ, i) * freqCounts[i];
		}
		int estimate = (int) Math.round(numVals + freqCounts[0] * numerSum / denomSum);
		return estimate < 1 ? 1 : estimate;
	}
}
