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

package org.apache.sysds.test.component.misc;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Hash;
import org.apache.sysds.utils.Hash.HashType;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class UtilHash {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		tests.add(new Object[] {100, 2, 0.0, 1.0});
		tests.add(new Object[] {100, 5, Double.MIN_VALUE, Double.MAX_VALUE});
		tests.add(new Object[] {1000, 50, Double.MIN_VALUE, Double.MAX_VALUE});
		tests.add(new Object[] {10000, 500, Double.MIN_VALUE, Double.MAX_VALUE});
		tests.add(new Object[] {1000, 500, Double.MIN_VALUE, Double.MAX_VALUE});
		tests.add(new Object[] {1000, 500, 0.0, 1.0});
		tests.add(new Object[] {1000, 500, 0.0, 100.0});
		tests.add(new Object[] {1000, 500, 0.0, 0.0000001});
		tests.add(new Object[] {1000, 1000, 0.0, 0.00000001});
		tests.add(new Object[] {1000000, 1000000, 0.0, 0.00000001});

		ArrayList<Object[]> actualTests = new ArrayList<>();

		Set<HashType> validHashTypes = new HashSet<>();
		for(HashType ht : HashType.values()) validHashTypes.add(ht);
		validHashTypes.remove(HashType.ExpHash);

		for(HashType ht : validHashTypes) {
			for(int i = 0; i < tests.size(); i++) {
				actualTests.add(new Object[] {tests.get(i)[0], tests.get(i)[1], tests.get(i)[2], tests.get(i)[3], ht});
			}
		}

		return actualTests;
	}

	@Parameterized.Parameter
	public int nrKeys = 1000;
	@Parameterized.Parameter(1)
	public int nrBuckets = 50;
	@Parameterized.Parameter(2)
	public double min;
	@Parameterized.Parameter(3)
	public double max;
	@Parameterized.Parameter(4)
	public HashType ht;

	private double epsilon = 0.05;

	@Test
	public void chiSquaredTest() {
		// https://en.wikipedia.org/wiki/Hash_function#Uniformity

		double[] input = TestUtils.generateTestMatrix(1, nrKeys, min, max, 1.0, 10)[0];

		int[] buckets = new int[nrBuckets];

		for(double x : input) {
			int hv = Hash.hash(new Double(x), ht);
			buckets[Math.abs(hv % nrBuckets)] += 1;
		}

		double top = 0;
		for(int b : buckets) {
			top += (double) (b) * (double) (b + 1.0) / 2.0;
		}

		double res = top / ((nrKeys / (2.0 * nrBuckets)) * (nrKeys + 2.0 * nrBuckets - 1));

		boolean success = Math.abs(res - 1) <= epsilon;

		Assert.assertTrue("Chi squared hashing test: " + res + " should be close to 1, with hashing: " + ht, success);
	}

}