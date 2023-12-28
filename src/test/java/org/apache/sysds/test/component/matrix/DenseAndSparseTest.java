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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibMatrixSparseToDense;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class DenseAndSparseTest {

	protected static final Log LOG = LogFactory.getLog(DenseAndSparseTest.class.getName());

	private final MatrixBlock a;
	private final int k;

	public DenseAndSparseTest(MatrixBlock a, int k) {
		this.a = a;
		this.k = k;
		LibMatrixSparseToDense.PAR_THRESHOLD = 100;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 1.0, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 1.0, 3), 4});

			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 0.1, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 0.1, 3), 4});

			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 0.001, 3), 4});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 0.001, 3), 4});

			// tests.add(new Object[] {forcedDense(1000, 1000, 0, 10, 0.001, 3), 4});
			// tests.add(new Object[] {forcedDense(1000, 1000, 0, 10, 0.1, 3), 4});
			// tests.add(new Object[] {forcedDense(1000, 1000, 0, 10, 0.01, 3), 4});
			// tests.add(new Object[] {forcedDense(1000, 1000, 0, 10, 0.02, 3), 4});
			// tests.add(new Object[] {forcedDense(10000, 100, 0, 10, 0.01, 3), 4});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static MatrixBlock forcedDense(int row, int col, double min, double max, double spar, int seed) {
		MatrixBlock x = TestUtils.generateTestMatrixBlock(row, col, min, max, spar, seed);
		x.sparseToDense();
		LOG.error(x);
		return x;
	}

	@Test
	public void toSparseCSR() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.denseToSparse(true);
		verify(a, b);
	}

	@Test
	public void toSparseMCSR() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.denseToSparse(false);
		verify(a, b);
	}

	@Test
	public void toSparseCSRUnknownNNZ() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.setNonZeros(-1);
		b.denseToSparse(true);
		verify(a, b);
	}

	@Test
	public void toSparseCSRIncorrectNNZ() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		if(b.getNonZeros() > 1) {
			b.setNonZeros(1);
			b.denseToSparse(true);
			verify(a, b);
		}
	}

	@Test
	public void toSparseMCSRUnknownNNZ() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.setNonZeros(-1);
		b.denseToSparse(false);
		verify(a, b);
	}

	@Test
	public void toDense() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, true);
		b.sparseToDense();
		verify(a, b);
	}

	@Test
	public void toSparseCSRParallel() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.denseToSparse(true, k);
		verify(a, b);
	}

	@Test
	public void toSparseMCSRParallel() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.denseToSparse(false, k);
		verify(a, b);
	}

	@Test
	public void toSparseCSRParallelUnknownNNZ() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.setNonZeros(-1);
		b.denseToSparse(true, k);

		verify(a, b);
	}

	@Test
	public void toSparseMCSRParallelUnknownNNZ() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, false);
		b.setNonZeros(-1);
		b.denseToSparse(false, k);
		verify(a, b);
	}

	@Test
	public void toDenseParallel() {
		MatrixBlock b = new MatrixBlock();
		b.copy(a, true);
		b.sparseToDense(k);
		verify(a, b);
	}

	private static void verify(MatrixBlock a, MatrixBlock b) {
		int n = a.getNumRows();
		int m = a.getNumColumns();
		assertEquals(n, b.getNumRows());
		assertEquals(m, b.getNumColumns());
		assertEquals(a.getNonZeros(), b.getNonZeros());
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				assertEquals(a.quickGetValue(i, j), b.quickGetValue(i, j), 0.000000);
			}
		}
	}
}
