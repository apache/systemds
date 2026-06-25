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
 * Tests the single-column (unweighted) branch of {@link MatrixBlock#pickValue(double, boolean)}.
 * The values are assumed to be sorted in ascending order, mirroring the contract used by the
 * quantile pick instructions. The two-column (weighted) branch is exercised separately through the
 * compressed sort tests.
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
		// pos = quantile * rlen; Math.round(pos), clamped to rlen-1.
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals(10, mb.pickValue(0.0, false), 0); // pos 0.0 -> idx 0
		assertEquals(20, mb.pickValue(0.2, false), 0); // pos 1.0 -> idx 1
		assertEquals(40, mb.pickValue(0.5, false), 0); // pos 2.5 -> round 3 -> idx 3
		assertEquals(50, mb.pickValue(1.0, false), 0); // pos 5.0 -> clamp idx 4
	}

	@Test
	public void pickOddLengthAverage() {
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		// pos 2.5 is non-integer -> average of floor(2.5)=2 and ceil(2.5)=3 -> (30+40)/2
		assertEquals(35, mb.pickValue(0.5, true), 0);
		// integer pos -> no averaging
		assertEquals(20, mb.pickValue(0.2, true), 0);
	}

	@Test
	public void pickEvenLengthAverage() {
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40}, false);
		// pos 1.5 -> average of idx 1 and idx 2 -> (20+30)/2
		assertEquals(25, mb.pickValue(0.375, true), 0);
		// pos 2.0 integer -> idx 2
		assertEquals(30, mb.pickValue(0.5, true), 0);
	}

	@Test
	public void pickAverageClampedAtTop() {
		// pos 4.75 -> ceil clamps to rlen-1, so the averaged pair is (idx4, idx4).
		MatrixBlock mb = singleColumn(new double[] {10, 20, 30, 40, 50}, false);
		assertEquals(50, mb.pickValue(0.95, true), 0);
	}

	@Test
	public void pickSparseSingleColumnWithZeros() {
		// Sorted ascending including leading zeros, stored sparse.
		MatrixBlock mb = singleColumn(new double[] {0, 0, 10, 20, 30}, true);
		assertEquals(0, mb.pickValue(0.0, false), 0); // pos 0.0 -> idx 0 (zero)
		assertEquals(20, mb.pickValue(0.5, false), 0); // pos 2.5 -> round 3 -> idx 3
		assertEquals(30, mb.pickValue(1.0, false), 0); // pos 5.0 -> clamp idx 4
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
