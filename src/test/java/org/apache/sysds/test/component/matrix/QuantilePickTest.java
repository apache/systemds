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
 * Tests the single-column (unweighted) branch of {@link MatrixBlock#pickValue(double, boolean)} and
 * {@link MatrixBlock#median()}. The values are assumed to be sorted in ascending order, mirroring the contract used
 * by the quantile pick instructions. The unweighted branch uses the same ceil-based rank as the two-column weighted
 * branch (with an implicit weight of 1 per value), so a single column yields the same quantile as the equivalent
 * (value, weight) representation. The two-column (weighted) branch is exercised separately through the compressed
 * sort tests.
 */
public class QuantilePickTest {

	private static MatrixBlock singleColumn(double[] values, boolean sparse) {
		MatrixBlock mb = new MatrixBlock(values.length, 1, sparse);
		for(int i = 0; i < values.length; i++)
			mb.set(i, 0, values[i]);
		mb.recomputeNonZeros();
		return mb;
	}

	@Test
	public void pickOddLengthNoAverage() {
		// rank = ceil(quantile * 5), value at (rank-1).
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals("q=0.0", 10, mb.pickValue(0.0, false), 0); // rank 0 -> idx 0
		assertEquals("q=0.2", 10, mb.pickValue(0.2, false), 0); // rank ceil(1.0)=1 -> idx 0
		assertEquals("q=0.5", 30, mb.pickValue(0.5, false), 0); // rank ceil(2.5)=3 -> idx 2
		assertEquals("q=0.75", 40, mb.pickValue(0.75, false), 0); // rank ceil(3.75)=4 -> idx 3
		assertEquals("q=1.0", 50, mb.pickValue(1.0, false), 0); // rank ceil(5.0)=5 -> idx 4
	}

	@Test
	public void pickOddLengthAverageSuppressed() {
		// Odd number of values -> averaging is suppressed, so average matches no-average.
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals("q=0.5 avg", 30, mb.pickValue(0.5, true), 0);
		assertEquals("q=0.75 avg", 40, mb.pickValue(0.75, true), 0);
	}

	@Test
	public void pickEvenLengthAverage() {
		// Even number of values -> averaging of adjacent order statistics applies.
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40}, false);
		assertEquals("q=0.25 avg", 15, mb.pickValue(0.25, true), 0); // rank 1 -> (idx0+idx1)/2
		assertEquals("q=0.375 avg", 25, mb.pickValue(0.375, true), 0); // rank ceil(1.5)=2 -> (idx1+idx2)/2
		assertEquals("q=0.5 avg", 25, mb.pickValue(0.5, true), 0); // rank 2 -> (idx1+idx2)/2
		assertEquals("q=0.75 avg", 35, mb.pickValue(0.75, true), 0); // rank 3 -> (idx2+idx3)/2
	}

	@Test
	public void pickEvenLengthNoAverage() {
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40}, false);
		assertEquals("q=0.25", 10, mb.pickValue(0.25, false), 0); // rank 1 -> idx 0
		assertEquals("q=0.5", 20, mb.pickValue(0.5, false), 0); // rank 2 -> idx 1
		assertEquals("q=0.75", 30, mb.pickValue(0.75, false), 0); // rank 3 -> idx 2
	}

	@Test
	public void pickAverageClampedAtTop() {
		// Top quantile: rank reaches the last element so there is no successor to average with.
		MatrixBlock even = singleColumn(new double[] {10, 20, 30, 40}, false);
		assertEquals("even q=0.95 avg", 40, even.pickValue(0.95, true), 0); // rank ceil(3.8)=4 -> idx 3, no avg
		assertEquals("even q=1.0 avg", 40, even.pickValue(1.0, true), 0);
		MatrixBlock odd = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals("odd q=0.95 avg", 50, odd.pickValue(0.95, true), 0); // odd -> avg suppressed
	}

	@Test
	public void pickSingleElement() {
		MatrixBlock mb = singleColumn(new double[] {42}, false);
		assertEquals("q=0.0", 42, mb.pickValue(0.0, false), 0);
		assertEquals("q=0.5", 42, mb.pickValue(0.5, false), 0);
		assertEquals("q=1.0", 42, mb.pickValue(1.0, false), 0);
		assertEquals("q=0.5 avg", 42, mb.pickValue(0.5, true), 0);
		assertEquals("median", 42, mb.median(), 0);
	}

	@Test
	public void pickSparseSingleColumnWithZeros() {
		// Sorted ascending including leading zeros, stored sparse.
		MatrixBlock mb = singleColumn(new double[] {0, 0, 10, 20, 30}, true);
		assertEquals("q=0.0", 0, mb.pickValue(0.0, false), 0); // rank 0 -> idx 0 (zero)
		assertEquals("q=0.5", 10, mb.pickValue(0.5, false), 0); // rank ceil(2.5)=3 -> idx 2
		assertEquals("q=0.75", 20, mb.pickValue(0.75, false), 0); // rank ceil(3.75)=4 -> idx 3
		assertEquals("q=1.0", 30, mb.pickValue(1.0, false), 0); // rank 5 -> idx 4
	}

	@Test
	public void medianSingleColumn() {
		// Odd length -> middle element; even length -> average of the two middle elements.
		assertEquals("odd median", 30, singleColumn(new double[] {10, 20, 30, 40, 50}, false).median(), 0);
		assertEquals("even median", 25, singleColumn(new double[] {10, 20, 30, 40}, false).median(), 0);
		assertEquals("sparse median", 10, singleColumn(new double[] {0, 0, 10, 20, 30}, true).median(), 0);
	}

	@Test
	public void pickSingleColumnMatchesDenseAndSparse() {
		double[] v = {-5, -1, 0, 2, 7, 9};
		MatrixBlock dense = singleColumn(v, false);
		MatrixBlock sparse = singleColumn(v, true);
		for(double q : new double[] {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0})
			for(boolean avg : new boolean[] {false, true})
				assertEquals("q=" + q + " avg=" + avg, dense.pickValue(q, avg), sparse.pickValue(q, avg), 0);
	}
}
