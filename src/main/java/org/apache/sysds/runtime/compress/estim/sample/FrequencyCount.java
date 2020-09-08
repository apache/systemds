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

import org.apache.sysds.runtime.compress.utils.ABitmap;

public class FrequencyCount {

	/**
	 * Creates an inverted histogram, where freqCounts[i-1] indicates how many values occurred with a frequency i. Note
	 * that freqCounts[0] represents the special values of the number of singletons.
	 * 
	 * @param ubm uncompressed bitmap
	 * @return frequency counts
	 */
	protected static int[] get(ABitmap ubm) {
		// determine max frequency
		int numVals = ubm.getNumValues();
		int maxCount = 0;
		for(int i = 0; i < numVals; i++)
			maxCount = Math.max(maxCount, ubm.getNumOffsets(i));

		// create frequency histogram
		int[] freqCounts = new int[maxCount];
		for(int i = 0; i < numVals; i++)
			freqCounts[ubm.getNumOffsets(i) - 1]++;

		return freqCounts;
	}
}
