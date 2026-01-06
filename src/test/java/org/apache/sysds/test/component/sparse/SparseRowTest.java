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

package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowScalar;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;


public class SparseRowTest extends AutomatedTestBase
{
	private final static int cols = 121;
	private final static int minVal = -10;
	private final static int maxVal = 10;
	private final static double sparsity = 0.3;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseRowEmptyToString()  {
		SparseRowScalar srs = new SparseRowScalar();
		assertEquals("", srs.toString());
	}

	@Test
	public void testSparseRowScalarInitZeroVal()  {
		SparseRowScalar srs = new SparseRowScalar(5, 0);
		srs.compact();
		assertEquals(-1, srs.getIndex());
	}

	@Test
	public void testSparseRowScalarSetNewVal()  {
		SparseRowScalar srs = new SparseRowScalar();
		assertTrue(srs.set(3, 5.0));
	}

	@Test
	public void testSparseRowScalarInvalidSet()  {
		SparseRowScalar srs = new SparseRowScalar(1, 1.0);
		RuntimeException ex = assertThrows(RuntimeException.class, () -> srs.set(3, 5.0));
		assertEquals("Invalid set to sparse row scalar.", ex.getMessage());
	}

	@Test
	public void testSparseRowScalarAppendZero()  {
		SparseRowScalar srs = new SparseRowScalar(1, 1.0);
		SparseRow srs2 = srs.append(2, 0.0);
		assertEquals(srs, srs2);
		assertNotEquals(0, srs2.values()[0]);
	}

	@Test
	public void testSparseRowScalarCompactZero()  {
		SparseRowScalar srs = new SparseRowScalar(1, 0.0);
		srs.compact();
		assertEquals(-1, srs.getIndex());
	}

	@Test
	public void testSparseRowScalarCompactNonZero()  {
		SparseRowScalar srs = new SparseRowScalar(1, 1.0);
		srs.compact();
		assertEquals(1, srs.getIndex());
	}

	@Test
	public void testSparseRowScalarCopy()  {
		SparseRowScalar srs = new SparseRowScalar(1, 1.0);
		SparseRowScalar srs2 = (SparseRowScalar) srs.copy(true);
		assertEquals(srs.getIndex(), srs2.getIndex());
		assertEquals(srs.getValue(), srs2.getValue(), 0.0);
		assertNotEquals(srs, srs2);
	}

	@Test
	public void testSparseRowVectorSetValues()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, sparsity, 7)[0];
		SparseRowVector srv = new SparseRowVector(UtilFunctions.computeNnz(v, 0, v.length), v, v.length);

		srv.compact();
		int nnz = srv.size();
		double[] w = getRandomMatrix(1, nnz, minVal, maxVal, 1, 13)[0];
		srv.setValues(w);

		assertArrayEquals(w, srv.values(), 0.0);
		assertEquals(srv.indexes().length, srv.values().length);
	}

	@Test
	public void testSparseRowVectorSetIndexes()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, 1, 7)[0];
		int nnz = UtilFunctions.computeNnz(v, 0, v.length);
		SparseRowVector srv = new SparseRowVector(nnz, v, v.length);

		int[] indexes = new int[nnz];
		for(int i = 0; i < nnz; i++) indexes[i] = i;
		srv.setIndexes(indexes);

		int idx = (int)(Math.random() * nnz);
		assertEquals(idx, srv.getIndex(idx));
		assertEquals(-1, srv.getIndex(nnz));
		assertEquals(srv.values().length, srv.indexes().length);
	}

	@Test
	public void testSparseRowVectorCopyFromLargerArray()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, sparsity, 7)[0];
		double[] w = getRandomMatrix(1, 2*cols, minVal, maxVal, sparsity, 7)[0];
		SparseRowVector srv = new SparseRowVector(UtilFunctions.computeNnz(v, 0, v.length), v, v.length);
		SparseRowVector other = new SparseRowVector(UtilFunctions.computeNnz(w, 0, w.length), w, w.length);
		srv.copy(other);

		assertArrayEquals(other.indexes(), srv.indexes());
		assertArrayEquals(other.values(), srv.values(), 0.0);
		assertNotEquals(other, srv);
	}

	@Test
	public void testSparseRowVectorSetEstimatedNzs()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, sparsity, 7)[0];
		int nnz = UtilFunctions.computeNnz(v, 0, v.length);
		SparseRowVector srv = new SparseRowVector(nnz, v, v.length);
		srv.setEstimatedNzs(nnz+1);
		assertEquals(nnz+1, srv.getEstimatedNzs());
	}

	@Test
	public void testSparseRowVectorSetAtPos()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, sparsity, 7)[0];
		int nnz = UtilFunctions.computeNnz(v, 0, v.length);
		SparseRowVector srv = new SparseRowVector(nnz, v, v.length);

		int pos = nnz-1;
		int col = 2;
		double val = 2.0;
		srv.setAtPos(pos, col, val);

		assertEquals(col, srv.indexes()[pos]);
		assertEquals(val, srv.indexes()[pos],0.0);
	}

	@Test
	public void testSparseRowVectorGetIndex()  {
		double[] v = getRandomMatrix(1, cols, minVal, maxVal, sparsity, 7)[0];
		int nnz = UtilFunctions.computeNnz(v, 0, v.length);
		SparseRowVector srv = new SparseRowVector(nnz, v, v.length);

		int pos = 0;
		srv.setAtPos(pos, 5, 2.0);
		int index = srv.getIndex(5);
		assertEquals(pos, index);

		int col2 = cols+1;
		int index2 = srv.getIndex(col2);
		assertEquals(-1, index2);
	}

	@Test
	public void testSparseRowVectorSearchIndexesFirstLTESizeZero()  {
		SparseRowVector srv = new SparseRowVector();
		int index = srv.searchIndexesFirstLTE(1);
		assertEquals(-1, index);
	}

	@Test
	public void testSparseRowVectorSearchIndexesFirstLTENotFound()  {
		SparseRowVector srv = new SparseRowVector(new double[] {1.0, 3.0}, new int[] {1, 3});
		int index = srv.searchIndexesFirstLTE(0);
		assertEquals(-1, index);
		int index2 = srv.searchIndexesFirstLTE(2);
		assertEquals(0, index2);
		int index3 = srv.searchIndexesFirstLTE(5);
		assertEquals(1, index3);
	}

	@Test
	public void testSparseRowVectorSetIndexRangeWithRecap()  {
		SparseRowVector srv = new SparseRowVector();
		srv.add(1, 1.0);
		srv.add(4, 4.0);
		srv.add(5, 5.0);
		srv.setIndexRange(2, 3, new double[]{2.0, 3.0}, 0, 2);
	}
}
