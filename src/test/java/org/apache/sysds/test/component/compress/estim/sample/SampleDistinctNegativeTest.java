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

import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory.EstimationType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class SampleDistinctNegativeTest {

	private final int[] frequencies;

	public SampleDistinctNegativeTest(int[] frequencies) {
		this.frequencies = frequencies;
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		tests.add(new Object[] {new int[] {-1}});
		tests.add(new Object[] {new int[] {-10}});

		tests.add(new Object[] {new int[] {-1022, 4, 2, 1, 3, -32}});
		tests.add(new Object[] {new int[] {10, 9, 8, 7, 6, 4, 4, 3, 2, 1, 0, -1}});
		
		// 0 is also invalid input of the frequency counts.
		// It is impossible to count 0 occurrences of anything.
		tests.add(new Object[] {new int[] {0}});

		return tests;
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testDistinctCountIsCorrectIfSampleIs100Percent() {
		SampleEstimatorFactory.distinctCount(frequencies, 100, 2, EstimationType.HassAndStokes);
	}

}
