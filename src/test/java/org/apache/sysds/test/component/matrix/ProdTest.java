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

public class ProdTest {

	private final MatrixBlock m;
	private final MatrixBlock mDense;

	public ProdTest() {
		m = new MatrixBlock(10, 10, true);

		m.allocateSparseRowsBlock();

		for(int i = 0; i < 10; i++)
			m.setValue(0, i, 3);

		mDense = new MatrixBlock(10, 10, false);
		mDense.copy(m);
		mDense.sparseToDense();
	}

	@Test
	public void testFullRowSparseProd() {
		// this test verifies that the prod call of a matrix where the sparse row is full, and the subsequent rows are
		// empty, returns 0.
		double ret = m.prod();
		assertEquals(ret, 0.0, 0.0);
	}

	@Test
	public void testFullProdIsEqualToDense() {
		double ret = m.prod();
		double retDense = mDense.prod();
		assertEquals(ret, retDense, 0.0);
	}

	@Test
	public void testZeroDimMatrix() {
		MatrixBlock mz = new MatrixBlock(0, 0, false);
		assertEquals(1, mz.prod(), 0.0);
	}

	@Test
	public void testZeroDimMatrixAllocateDense() {
		MatrixBlock mz = new MatrixBlock(0, 0, false);
		mz.allocateDenseBlock();
		assertEquals(1, mz.prod(), 0.0);
	}

	@Test
	public void testZeroDimMatrixAllocateSparse() {
		MatrixBlock mz = new MatrixBlock(0, 0, false);
		mz.allocateSparseRowsBlock();
		assertEquals(1, mz.prod(), 0.0);
	}

	@Test
	public void testEmptyProd() {
		MatrixBlock empty = new MatrixBlock(10, 10, false);
		assertEquals(0, empty.prod(), 0.0);
	}
}
