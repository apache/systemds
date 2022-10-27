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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.matrix.data.LibMatrixEquals;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class EqualsTest {

	@Test
	public void sameObject() {
		MatrixBlock mb = new MatrixBlock();
		assertTrue(mb.equals(mb));
	}

	@Test
	public void sameObject2() {
		MatrixBlock mb = new MatrixBlock(10, 10, -2.0);
		assertTrue(mb.equals(mb));
	}

	@Test
	public void sameObjectInterface() {
		MatrixBlock mb = new MatrixBlock(10, 10, -2.0);
		assertTrue(LibMatrixEquals.equals(mb, mb));
	}

	@Test
	public void sameObjectInterface2() {
		MatrixBlock mb = new MatrixBlock(10, 10, -2.0);
		assertTrue(LibMatrixEquals.equals(mb, mb, 1.0));
	}

	@Test
	public void sameIfConstructedEqual_Empty() {
		MatrixBlock mb1 = new MatrixBlock();
		MatrixBlock mb2 = new MatrixBlock();
		assertTrue(mb1.equals(mb2));
		assertTrue(mb2.equals(mb1));
	}

	@Test
	public void sameIfConstructedEqual_CastToObject() {
		MatrixBlock mb1 = new MatrixBlock();
		MatrixBlock mb2 = new MatrixBlock();
		assertTrue(mb1.equals((Object) mb2));
		assertTrue(mb2.equals((Object) mb1));
	}

	@Test
	public void oneIsEmpty() {
		MatrixBlock empty = new MatrixBlock(10, 10, 0.0);
		MatrixBlock full = new MatrixBlock(10, 10, 1.0);

		assertFalse(empty.equals(full));
		assertFalse(full.equals(empty));
	}

	@Test
	public void diffRows() {

		MatrixBlock empty = new MatrixBlock(11, 10, 0.0);
		MatrixBlock empty2 = new MatrixBlock(10, 10, 0.0);

		assertFalse(empty.equals(empty2));
		assertFalse(empty2.equals(empty));
	}

	@Test
	public void diffCols() {

		MatrixBlock empty = new MatrixBlock(10, 10, 0.0);
		MatrixBlock empty2 = new MatrixBlock(10, 11, 0.0);

		assertFalse(empty.equals(empty2));
		assertFalse(empty2.equals(empty));
	}

	@Test
	public void diffRowsAndCols() {

		MatrixBlock empty = new MatrixBlock(13, 10, 0.0);
		MatrixBlock empty2 = new MatrixBlock(10, 11, 0.0);

		assertFalse(empty.equals(empty2));
		assertFalse(empty2.equals(empty));
	}

	@Test
	public void diffNNZ() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setValue(1, 1, 3);
		m1.setValue(1, 2, 3);

		m2.setValue(1, 1, 3);
		m2.setValue(1, 2, 3);
		m2.setValue(1, 3, 3);

		assertFalse(m1.equals(m2));
		assertFalse(m2.equals(m1));
	}

	@Test
	public void unknownNNZ_m1() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setValue(1, 1, 3);
		m1.setValue(1, 2, 3);

		m1.setNonZeros(-1);

		m2.setValue(1, 1, 3);
		m2.setValue(1, 2, 3);
		m2.setValue(1, 3, 3);

		assertFalse(m1.equals(m2));
		assertFalse(m2.equals(m1));
	}

	@Test
	public void unknownNNZ_m2() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setValue(1, 1, 3);
		m1.setValue(1, 2, 3);

		m2.setValue(1, 1, 3);
		m2.setValue(1, 2, 3);
		m2.setValue(1, 3, 3);

		m2.setNonZeros(-1);

		assertFalse(m1.equals(m2));
		assertFalse(m2.equals(m1));
	}

	@Test
	public void unknownNNZboth() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setValue(1, 1, 3);
		m1.setValue(1, 2, 3);

		m1.setNonZeros(-1);

		m2.setValue(1, 1, 3);
		m2.setValue(1, 2, 3);
		m2.setValue(1, 3, 3);

		m2.setNonZeros(-1);

		assertFalse(m1.equals(m2));
		assertFalse(m2.equals(m1));
	}

	@Test
	public void unknownNNZEmptyBoth() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setNonZeros(-1);
		m2.setNonZeros(-1);

		assertTrue(m1.equals(m2));
		assertTrue(m2.equals(m1));
	}

	@Test
	public void unknownNNZEmptyOne() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 0.0);

		m1.setNonZeros(-1);

		assertTrue(m1.equals(m2));
		assertTrue(m2.equals(m1));
	}

	@Test
	public void unknownNNZEmptyOneFull() {

		MatrixBlock m1 = new MatrixBlock(10, 10, 0.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 1.0);

		m2.setNonZeros(-1);

		assertFalse(m1.equals(m2));
		assertFalse(m2.equals(m1));
	}

	@Test
	public void equivalentWithVeryVerySmallEps() {
		MatrixBlock m1 = new MatrixBlock(10, 10, 1.0 + Double.MIN_VALUE * 10);
		MatrixBlock m2 = new MatrixBlock(10, 10, 1.0);
		assertTrue(m1.equals(m2));
		assertTrue(m2.equals(m1));
	}

	@Test
	public void equivalentWithEpsSet() {
		MatrixBlock m1 = new MatrixBlock(10, 10, 1.0 + 1.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 1.0);
		assertTrue(LibMatrixEquals.equals(m1, m2, 1.0));
		assertTrue(LibMatrixEquals.equals(m2, m1, 1.0));
	}

	@Test
	public void notEquivalentWithEpsSet() {
		MatrixBlock m1 = new MatrixBlock(10, 10, 1.0 + 1.0);
		MatrixBlock m2 = new MatrixBlock(10, 10, 1.0);
		assertFalse(LibMatrixEquals.equals(m1, m2, 0.9999));
		assertFalse(LibMatrixEquals.equals(m2, m1, 0.9999));
	}

	@Test
	public void testSparse() {

		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 1000, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 1000, 0, 100, 0.05, 231);
		assertTrue(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertTrue(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

	@Test
	public void testSparseOneForcedDense() {

		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);
		m1.sparseToDense();
		assertTrue(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertTrue(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

	@Test
	public void testSparseOneForcedDenseNotEquivalent() {

		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);

		m1.getSparseBlock().get(13).values()[2] = 1324;
		m2.sparseToDense();

		assertFalse(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertFalse(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

	@Test
	public void testSparseNotEquivalent() {

		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 1000, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(100, 1000, 0, 100, 0.05, 231);

		m1.getSparseBlock().get(13).values()[2] = 1324;

		assertFalse(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertFalse(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

	@Test
	public void testForcedNonContiguousNotEqual() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.mockNonContiguousMatrix( //
			TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231));
		m1.getSparseBlock().get(13).values()[2] = 1324;
		assertFalse(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertFalse(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

	@Test
	public void testForcedNonContiguousEqual() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231);
		MatrixBlock m2 = TestUtils.mockNonContiguousMatrix( //
			TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.05, 231));
		assertTrue(LibMatrixEquals.equals(m1, m2, 0.00000001));
		assertTrue(LibMatrixEquals.equals(m2, m1, 0.00000001));
	}

}
