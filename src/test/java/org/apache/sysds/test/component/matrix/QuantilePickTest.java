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

package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

/**
 * Tests the single-column (unweighted) branch of {@link MatrixBlock#pickValue(double)} and
 * {@link MatrixBlock#median()}. The values are assumed to be sorted in ascending order. Results follow R's default
 * quantile definition (type 7): linear interpolation between adjacent order statistics with h = (n - 1) * p + 1, g = h
 * - floor(h), Q = (1 - g) * x[floor(h)] + g * x[floor(h) + 1].
 */
public class QuantilePickTest {

	private static final double EPS = 1e-12;

	private static MatrixBlock singleColumn(double[] values, boolean sparse) {
		MatrixBlock mb = new MatrixBlock(values.length, 1, sparse);
		for(int i = 0; i < values.length; i++)
			mb.set(i, 0, values[i]);
		mb.recomputeNonZeros();
		return mb;
	}

	@Test
	public void pickOddLength() {
		// n=5: h = 4p + 1. At classical quantiles the rank is integer, so no interpolation is needed.
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals("q=0.0", 10, mb.pickValue(0.0), EPS);
		assertEquals("q=0.2", 18, mb.pickValue(0.2), EPS); // h=1.8, 0.2*10 + 0.8*20
		assertEquals("q=0.25", 20, mb.pickValue(0.25), EPS); // h=2 -> x[2]=20
		assertEquals("q=0.5", 30, mb.pickValue(0.5), EPS); // h=3 -> x[3]=30
		assertEquals("q=0.75", 40, mb.pickValue(0.75), EPS); // h=4 -> x[4]=40
		assertEquals("q=1.0", 50, mb.pickValue(1.0), EPS);
	}

	@Test
	public void pickEvenLength() {
		// n=4: h = 3p + 1. Classical p != 0.5 land between order statistics and interpolate.
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40}, false);
		assertEquals("q=0.25", 17.5, mb.pickValue(0.25), EPS); // h=1.75 -> 0.25*10 + 0.75*20
		assertEquals("q=0.375", 21.25, mb.pickValue(0.375), EPS); // h=2.125 -> 0.875*20 + 0.125*30
		assertEquals("q=0.5", 25, mb.pickValue(0.5), EPS); // h=2.5 -> 0.5*20 + 0.5*30
		assertEquals("q=0.75", 32.5, mb.pickValue(0.75), EPS); // h=3.25 -> 0.75*30 + 0.25*40
	}

	@Test
	public void pickClampedAtTop() {
		// Top quantile is clamped so no successor is required for interpolation.
		MatrixBlock even = singleColumn(new double[] {10, 20, 30, 40}, false);
		assertEquals("even q=0.95", 38.5, even.pickValue(0.95), EPS); // h=3.85 -> 0.15*30 + 0.85*40
		assertEquals("even q=1.0", 40, even.pickValue(1.0), EPS);
		MatrixBlock odd = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals("odd q=0.95", 48, odd.pickValue(0.95), EPS); // h=4.8 -> 0.2*40 + 0.8*50
	}

	@Test
	public void pickSingleElement() {
		MatrixBlock mb = singleColumn(new double[] {42}, false);
		assertEquals("q=0.0", 42, mb.pickValue(0.0), EPS);
		assertEquals("q=0.5", 42, mb.pickValue(0.5), EPS);
		assertEquals("q=1.0", 42, mb.pickValue(1.0), EPS);
		assertEquals("median", 42, mb.median(), EPS);
	}

	@Test
	public void pickSparseSingleColumnWithZeros() {
		// Sorted ascending including leading zeros, stored sparse. n=5, h = 4p + 1.
		MatrixBlock mb = singleColumn(new double[] {0, 0, 10, 20, 30}, true);
		assertEquals("q=0.0", 0, mb.pickValue(0.0), EPS);
		assertEquals("q=0.25", 0, mb.pickValue(0.25), EPS); // h=2 -> x[2]=0
		assertEquals("q=0.5", 10, mb.pickValue(0.5), EPS); // h=3 -> x[3]=10
		assertEquals("q=0.75", 20, mb.pickValue(0.75), EPS); // h=4 -> x[4]=20
		assertEquals("q=1.0", 30, mb.pickValue(1.0), EPS); // h=5 -> x[5]=30
	}

	@Test
	public void medianSingleColumn() {
		assertEquals("odd median", 30, singleColumn(new double[] {10, 20, 30, 40, 50}, false).median(), EPS);
		assertEquals("even median", 25, singleColumn(new double[] {10, 20, 30, 40}, false).median(), EPS);
		assertEquals("sparse median", 10, singleColumn(new double[] {0, 0, 10, 20, 30}, true).median(), EPS);
	}

	@Test
	public void pickSingleColumnMatchesDenseAndSparse() {
		double[] v = {-5, -1, 0, 2, 7, 9};
		MatrixBlock dense = singleColumn(v, false);
		MatrixBlock sparse = singleColumn(v, true);
		for(double q : new double[] {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0})
			assertEquals("q=" + q, dense.pickValue(q), sparse.pickValue(q), EPS);
	}
}
