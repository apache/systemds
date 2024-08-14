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

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)

public class ReshapeTest {
	protected static final Log LOG = LogFactory.getLog(ReshapeTest.class.getName());

	@Parameterized.Parameter
	public int k;
	@Parameterized.Parameter(1)
	public boolean rowWise;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		tests.add(new Object[] {1, true});
		tests.add(new Object[] {10, true});
		tests.add(new Object[] {-1, true});
		tests.add(new Object[] {1, false});
		tests.add(new Object[] {10, false});
		tests.add(new Object[] {-1, false});

		return tests;
	}

	@Test
	public void reshapeDense1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 5, 10);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 5, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 5, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 24, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 24, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(2, 2, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 1, 4);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense5() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 25, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense6() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 25, 4);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense7() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 25, 4);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense8() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		DenseBlock db = a.getDenseBlock();
		DenseBlock spy = spy(db);
		when(spy.numBlocks()).thenReturn(2);
		a.setDenseBlock(spy);
		MatrixBlock b = runReshape(a, 25, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense9() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(25, 4, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 100, 1);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense10() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(25, 4, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 1, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense11() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(25, 4, 0, 10, 1.0, 132);
		a.setDenseBlock(null);
		MatrixBlock b = runReshape(a, 1, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4OtherInterface() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = LibMatrixReorg.reshape(a, null, 25, 200, rowWise);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4OtherInterfaceObject() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = LibMatrixReorg.reshape(a, new MatrixBlock(), 25, 200, rowWise);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSameSize1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 100, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSameSize2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		MatrixBlock b = runReshape(a, 10, 5);
		verifyEqualReshaped(a, b);
	}

	@Test(expected = Exception.class)
	public void invalid1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		runReshape(a, 10, 10);
	}

	@Test(expected = Exception.class)
	public void invalid2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		runReshape(a, 5, 5);
	}

	@Test(expected = Exception.class)
	public void invalid3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		runReshape(a, 12, 12);
	}

	@Test
	public void reshapeSparseToDense() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 100, 1);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 50, 2);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.9, 132);

		a.denseToSparse(true);
		MatrixBlock b = runReshape(a, 200, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.9, 132);
		a.denseToSparse(true);
		MatrixBlock b = runReshape(a, 50, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense5() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.9, 132);
		a.getDenseBlock().fillRow(3, 0);
		a.denseToSparse(true);
		MatrixBlock b = runReshape(a, 50, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense6() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(3, 100, 0, 10, 1.0, 132);
		a.getDenseBlock().fillRow(1, 0);
		a.denseToSparse(true);
		MatrixBlock b = runReshape(a, 1, 300);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense7() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 1.0, 132);
		a.getDenseBlock().fillRow(0, 0);
		a.denseToSparse(false);
		MatrixBlock b = runReshape(a, 2, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 1, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 2, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 2, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 2, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 2, 0, 10, 0.1, 132);
		a.setDenseBlock(null);
		MatrixBlock b = runReshape(a, 2, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse5() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(50, 2, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 1, 100);
		verifyEqualReshaped(a, b);
	}


	@Test
	public void reshapeDenseToSparse6() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		a.sparseToDense();
		MatrixBlock b = runReshape(a, 2, 50);
		verifyEqualReshaped(a, b);
	}


	@Test
	public void reshapeSparseToSparse() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 50, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 25, 400);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 25, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 1, 100 * 125);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse5() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 132);
		MatrixBlock b = runReshape(a, 1, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse6() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.001, 132);
		MatrixBlock b = runReshape(a, 50, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse7() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 100, 0, 10, 0.0001, 132);
		MatrixBlock b = runReshape(a, 1, 1000);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse8() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 1000, 0, 10, 0.0001, 132);
		MatrixBlock b = runReshape(a, 2, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse9() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 1000, 0, 10, 0.001, 132);
		MatrixBlock b = runReshape(a, 2, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse10() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(4, 250, 0, 10, 0.001, 132);
		a.setSparseBlock(new SparseBlockCSR(a.getSparseBlock()));
		MatrixBlock b = runReshape(a, 2, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse11() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(4, 250, 0, 10, 0.001, 132);
		MatrixBlock spy = spy(a);
		when(spy.getNonZeros()).thenReturn((long) Integer.MAX_VALUE + 45L);
		MatrixBlock b = runReshape(spy, 2, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse12() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(4, 125, 0, 10, (double)1 / (125*2), 132);
		MatrixBlock b = runReshape(a, 1, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse13() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		a.getSparseBlock().reset(0, 10, 100);
		MatrixBlock b = runReshape(a, 2, 50);
		verifyEqualReshaped(a, b);
	}


	@Test
	public void reshapeEmpty1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.0, 132);
		MatrixBlock b = runReshape(a, 25, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 0.0, 132);
		MatrixBlock b = runReshape(a, 5, 20);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(5, 5, 0, 10, 0.0, 132);
		MatrixBlock b = runReshape(a, 1, 25);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(5, 5, 0, 10, 0.0, 132);
		MatrixBlock b = runReshape(a, 25, 1);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty5() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(6, 6, 0, 10, 0.0, 132);
		MatrixBlock b = runReshape(a, 3, 12);
		verifyEqualReshaped(a, b);
	}

	private MatrixBlock runReshape(MatrixBlock a, int r, int c) {
		return LibMatrixReorg.reshape(a, null, r, c, rowWise, k);
	}

	private void verifyEqualReshaped(MatrixBlock expected, MatrixBlock actual) {
		final long expectedCells = (long) expected.getNumRows() * expected.getNumColumns();
		final long actualCells = (long) actual.getNumRows() * actual.getNumColumns();

		assertEquals(expectedCells, actualCells);
		final long eCols = expected.getNumColumns();
		final long eRows = expected.getNumRows();
		final long aCols = actual.getNumColumns();
		final long aRows = actual.getNumRows();

		if(rowWise) {

			for(long c = 0; c < expectedCells; c++) {
				int r1 = (int) (c / eCols);
				int c1 = (int) (c % eCols);

				int r2 = (int) (c / aCols);
				int c2 = (int) (c % aCols);
				if(expected.get(r1, c1) != actual.get(r2, c2)) {
					double v1 = expected.get(r1, c1);
					double v2 = actual.get(r2, c2);
					String err = String.format("%d,%d vs %d,%d not equal with values: %f vs %f", r1, c1, r2, c2, v1, v2);
					assertEquals(err, v1, v2, 0.0);
				}
			}
		}
		else {

			for(long c = 0; c < expectedCells; c++) {
				int r2 = (int) (c / aCols);
				int c2 = (int) (c % aCols);

				int r1 = (int) ((aRows * c2 + r2) % eRows);
				int c1 = (int) ((aRows * c2 + r2) / eRows);

				if(expected.get(r1, c1) != actual.get(r2, c2)) {
					double v1 = expected.get(r1, c1);
					double v2 = actual.get(r2, c2);
					String err = String.format("%d,%d vs %d,%d not equal with values: %f vs %f", r1, c1, r2, c2, v1, v2);
					assertEquals(err, v1, v2, 0.0);
				}
			}
		}

	}
}
