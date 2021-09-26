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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;

public class SampleEstimatorFactory {

	protected static final Log LOG = LogFactory.getLog(SampleEstimatorFactory.class.getName());

	public enum EstimationType {
		HassAndStokes, ShlosserEstimator, ShlosserJackknifeEstimator, SmoothedJackknifeEstimator,
	}

	public static int distinctCount(int[] frequencies, int nRows, int sampleSize, EstimationType type,
		HashMap<Integer, Double> solveCache) {
		final int numVals = frequencies.length;
		try {
			int[] invHist = getInvertedFrequencyHistogram(frequencies);
			switch(type) {
				case HassAndStokes:
					return HassAndStokes.distinctCount(numVals, invHist, nRows, sampleSize, solveCache);
				case ShlosserEstimator:
					return ShlosserEstimator.distinctCount(numVals, invHist, nRows, sampleSize);
				case ShlosserJackknifeEstimator:
					return ShlosserJackknifeEstimator.distinctCount(numVals, frequencies, invHist, nRows, sampleSize);
				case SmoothedJackknifeEstimator:
					return SmoothedJackknifeEstimator.distinctCount(numVals, invHist, nRows, sampleSize);
				default:
					throw new NotImplementedException("Type not yet supported for counting distinct: " + type);
			}
		}
		catch(Exception e) {
			throw new DMLCompressionException(
				"Error while estimating distinct count with arguments:\n" + numVals + " frequencies:\n"
					+ Arrays.toString(frequencies) + "\n nrows: " + nRows + " " + sampleSize + " type: " + type,
				e);
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
