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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ReshapeTest {
	protected static final Log LOG = LogFactory.getLog(ReshapeTest.class.getName());

	@Test
	public void reshapeDense1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 5, 10);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 5, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 5, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 24, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 24, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 25, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4OtherInterface() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = LibMatrixReorg.reshape(a, null, 25, 200, true);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDense4OtherInterfaceObject() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = LibMatrixReorg.reshape(a, new MatrixBlock(), 25, 200, true);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSameSize1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 50, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 100, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSameSize2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		MatrixBlock b = reshapeSingle(a, 10, 5);
		verifyEqualReshaped(a, b);
	}

	@Test(expected = Exception.class)
	public void invalid1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		reshapeSingle(a, 10, 10);
	}

	@Test(expected = Exception.class)
	public void invalid2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		reshapeSingle(a, 5, 5);
	}

	@Test(expected = Exception.class)
	public void invalid3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 132);
		reshapeSingle(a, 12, 12);
	}

	@Test
	public void reshapeSparseToDense() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 100, 1);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToDense2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 50, 2);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 1, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 2, 50);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeDenseToSparse3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 2, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 2, 100);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 50, 200);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 25, 400);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse3() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 25, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeSparseToSparse4() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.1, 132);
		MatrixBlock b = reshapeSingle(a, 1, 100 * 125);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty1() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 125, 0, 10, 0.0, 132);
		MatrixBlock b = reshapeSingle(a, 25, 500);
		verifyEqualReshaped(a, b);
	}

	@Test
	public void reshapeEmpty2() {
		MatrixBlock a = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 0.0, 132);
		MatrixBlock b = reshapeSingle(a, 5, 20);
		verifyEqualReshaped(a, b);
	}

	private MatrixBlock reshapeSingle(MatrixBlock a, int r, int c) {
		return LibMatrixReorg.reshape(a, r, c, true);
	}

	private void verifyEqualReshaped(MatrixBlock expected, MatrixBlock actual) {
		final long expectedCells = (long) expected.getNumRows() * expected.getNumColumns();
		final long actualCells = (long) actual.getNumRows() * actual.getNumColumns();

		assertEquals(expectedCells, actualCells);

		final long eCols = expected.getNumColumns();
		final long aCols = actual.getNumColumns();

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
}
