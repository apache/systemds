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

package org.apache.sysds.test.component.compress.estim.sample;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.estim.sample.ShlosserEstimator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class ShlosserEstimatorTest {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		final int m = Integer.MAX_VALUE;
		tests.add(create(new int[] {0, 0, 0, 0, 0, m, m, m, m, m}));
		tests.add(create(new int[] {m, m, m, m, m}));
		tests.add(create(new int[] {m, m, m, m}));
		tests.add(create(new int[] {m, m, m}));
		tests.add(create(new int[] {m, m}));
		tests.add(create(new int[] {m}));

		final int l = Integer.MIN_VALUE;
		tests.add(create(new int[] {l}));

		tests.add(create(new int[] {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m}));


		tests.add(createHardBigSample(new int[]{1,0,0,0, m}));


		tests.add(createHardSmallSample(new int[]{1,0,0,0, 99}));

		return tests;
	}

	private static Object[] create(int[] frequencies) {
		return new Object[] {frequencies, 100000L, 10000000, 9999999};
	}

	private static Object[] createHardBigSample(int[] frequencies) {
		return new Object[] {frequencies, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE-1};
	}

	private static Object[] createHardSmallSample(int[] frequencies) {
		return new Object[] {frequencies, Long.MAX_VALUE, Long.MAX_VALUE, 100};
	}

	final long numVals;
	final int[] freqCounts;
	final long nRows;
	final long sampleSize;

	public ShlosserEstimatorTest(int[] freqCounts, long numVals, long nRows, long sampleSize) {
		this.freqCounts = freqCounts;
		this.numVals = numVals;
		this.nRows = nRows;
		this.sampleSize = sampleSize;
	}

	@Test
	public void testWildEstimates() {
		ShlosserEstimator.distinctCount(numVals, freqCounts, nRows, sampleSize);
	}

}
