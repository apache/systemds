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

package org.apache.sysds.test.component.compress.estim;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory.EstimationType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class SampleDistinctTest {

	private final int[] frequencies;
	private final int total;
	private final EstimationType type;
	private final HashMap<Integer, Double> solveCache;

	public SampleDistinctTest(int[] frequencies, EstimationType type, HashMap<Integer, Double> solveCache) {
		this.frequencies = frequencies;
		this.type = type;
		this.solveCache = solveCache;
		int t = 0;
		if(frequencies != null)
			for(int f : frequencies)
				t += f;
		total = t;
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		HashMap<Integer, Double> solveCache = new HashMap<>();

		for(EstimationType type : EstimationType.values()) {
			tests.add(new Object[] {null, type, solveCache});
			tests.add(new Object[] {new int[] {}, type, solveCache});
			tests.add(new Object[] {new int[] {97, 6, 56, 4, 242, 123, 2}, type, solveCache});
			tests.add(new Object[] {new int[] {6, 5}, type, solveCache});
			tests.add(new Object[] {new int[] {2, 1, 1, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {5, 4, 2, 2, 1, 1, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {7, 7, 7, 7, 6, 5, 4, 4, 3, 3, 2, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {413, 37, 20, 37, 32, 37, 4, 17, 1, 3, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {414, 37, 20, 37, 32, 37, 4, 17, 1, 3, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {415, 37, 20, 37, 32, 37, 4, 17, 1, 3, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {416, 37, 20, 37, 32, 37, 4, 17, 1, 3, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {417, 37, 20, 37, 32, 37, 4, 17, 1, 3, 1, 1, 1}, type, solveCache});

			tests.add(new Object[] {new int[] {1, 1, 1, 1, 1, 1, 1, 1, 1}, type, solveCache});
			tests.add(new Object[] {new int[] {500, 500, 500, 500}, type, solveCache});
			tests.add(new Object[] {new int[] {500, 400, 300, 200}, type, solveCache});
			tests.add(new Object[] {new int[] {1000, 400, 300, 200}, type, solveCache});
			tests.add(new Object[] {new int[] {1000, 400, 300, 200, 2, 2, 2, 2, 4, 2, 13, 3, 2, 1, 4, 2, 3, 2, 2, 2, 2, 2,
				2, 2, 1, 1, 1, 1, 1, 3, 4, 2, 1, 3, 2}, type, solveCache});
			tests.add(new Object[] {new int[] {1000, 400, 300, 200, 2, 2, 2, 2, 4, 2, 13, 3, 2, 1, 4, 2, 3, 2, 2, 2, 2, 2,
				2, 2, 1, 1, 1, 1, 1, 3, 4, 2, 1, 3, 2, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
				10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}, type, solveCache});

			tests.add(new Object[] {
				new int[] {1500, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
					9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1},
				type, solveCache});

			for(int i = 1; i < 10; i++) {
				tests.add(new Object[] {new int[] {i, i, i, i, i}, type, solveCache});
				tests.add(new Object[] {new int[] {i, i + 1, i + 2, i + 3, i + 4}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 1, 1, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 2, 1, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 2, 2, 2, 2, 1, 1, 1, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 2, 2, 1, 1, 1, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 2, 2, 1, 1, 1, 1}, type, solveCache});
				tests.add(new Object[] {new int[] {i, 2, 1, 1, 1, 1, 1}, type, solveCache});
			}
			tests.add(new Object[] {new int[] {8, 5, 3, 2, 2, 2, 2}, type, solveCache});
			tests.add(new Object[] {new int[] {8, 5, 3, 2, 2, 2, 2}, type, solveCache});

		}

		// Fuzzing test.
		// Random r = new Random();
		// for(int i = 0; i < 10000; i++) {
		// tests.add(new Object[] {new int[] {r.nextInt(10) + 1, r.nextInt(10) + 1, r.nextInt(10) + 1, r.nextInt(10) + 1,
		// r.nextInt(10) + 1, r.nextInt(10) + 1}, EstimationType.HassAndStokes, solveCache});
		// tests.add(new Object[] {new int[] {r.nextInt(100) + 1, r.nextInt(100) + 1, r.nextInt(100) + 1,
		// r.nextInt(100) + 1, r.nextInt(100) + 1, r.nextInt(100) + 1}, EstimationType.HassAndStokes, solveCache});
		// tests.add(new Object[] {new int[] {r.nextInt(10) + 1, r.nextInt(10) + 1, r.nextInt(10) + 1, r.nextInt(10) + 1,
		// r.nextInt(10) + 1, r.nextInt(10) + 1, 1}, EstimationType.HassAndStokes, solveCache});
		// tests.add(new Object[] {new int[] {r.nextInt(100) + 1, r.nextInt(100) + 1, r.nextInt(100) + 1,
		// r.nextInt(100) + 1, r.nextInt(100) + 1, r.nextInt(100) + 1, 1}, EstimationType.HassAndStokes, solveCache});
		// }

		return tests;
	}

	@Test
	public void testDistinctCountIsCorrectIfSampleIs100Percent() {
		// Sample 100%
		int nRows = total;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 1.0);
	}

	@Test
	public void test20p() {
		// Sample 20%
		int nRows = total * 5;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 0.2);
	}

	@Test
	public void test1p() {
		// Sample 1%
		int nRows = total * 100;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 0.01);
	}

	@Test
	public void test01p() {
		// Sample 0.1%
		int nRows = total * 1000;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 0.001);
	}

	@Test
	public void test001p() {
		// Sample 0.01%
		int nRows = total * 10000;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 0.0001);
	}

	@Test
	public void test0001p() {
		// Sample 0.001%
		int nRows = total * 100000;
		int sampleSize = total;
		int c = SampleEstimatorFactory.distinctCount(frequencies, nRows, sampleSize, type, solveCache);
		verify(c, 0.00001);
	}

	private void verify(int c, double p) {
		if(frequencies == null)
			assertEquals(0, c);
		else if(p == 1.0 && frequencies.length != c) {
			String m = "incorrect estimate with type; " + type + " est: " + c + " frequencies: "
				+ Arrays.toString(frequencies);
			assertEquals(m, frequencies.length, c);
		}
		else if(c < frequencies.length)
			fail("estimate is lower than observed elements");
		else if(c > Math.ceil((double) total / p) - frequencies.length + total)
			fail("estimate " + c + " is larger than theoretical max uniques "
				+ (Math.ceil((double) total / p) - frequencies.length + total) + " using: " + type);
	}
}
