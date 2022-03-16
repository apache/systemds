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

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class EigenDecompTest {

	protected static final Log LOG = LogFactory.getLog(EigenDecompTest.class.getName());

	@Test
	public void testLanczosSimple() {
		MatrixBlock in = new MatrixBlock(4, 4, false);
		// 4, 1, -2, 2
		// 1, 2, 0, 1
		// -2, 0, 3, -2
		// 2, 1, -2, -1
		double[] a = {4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1};
		in.init(a, 4, 4);
		testLanczos(in, 1e-4, 1);
	}

	@Test
	public void testLanczosSmall() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testLanczos(in, 1e-4, 1);
	}

	@Test
	public void testLanczosLarge() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(100, 100, 0.0, 1.0, 1.0, 1);
		testLanczos(in, 1e-4, 1);
	}

	@Test
	public void testLanczosLargeMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(100, 100, 0.0, 1.0, 1.0, 1);
		testLanczos(in, 1e-4, threads);
	}

	private void testLanczos(MatrixBlock in, double tol, int threads) {
		try {
			long t1 = System.nanoTime();
			MatrixBlock[] m1 = LibCommonsMath.multiReturnOperations(in, "eigen", threads, 1);

			long t2 = System.nanoTime();
			MatrixBlock[] m2 = LibCommonsMath.multiReturnOperations(in, "eigen_lanczos", threads, 1);

			long t3 = System.nanoTime();

			LOG.error("time eigen: " + (t2 - t1) + " time Lanczos: " + (t3 - t2) + " Lanczos speedup: "
				+ ((double) (t2 - t1) / (t3 - t2)));
			TestUtils.compareMatrices(m1[0], m2[0], tol, "Result of eigenvalues of new eigen_lanczos function wrong");
			testEvecValues(m1[1], m2[1], tol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testQREigenSimple() {
		MatrixBlock in = new MatrixBlock(4, 4, false);
		// 52, 30, 49, 28
		// 30, 50, 8, 44
		// 49, 8, 46, 16
		// 28, 44, 16, 22
		double[] a = {52, 30, 49, 28, 30, 50, 8, 44, 49, 8, 46, 16, 28, 44, 16, 22};
		in.init(a, 4, 4);
		testQREigen(in, 1e-4, 1);
	}

	@Test
	public void testQREigenSymSmall() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testQREigen(in, 1e-4, 1);
	}

	@Test
	public void testQREigenSymSmallMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testQREigen(in, 1e-4, threads);
	}

	@Test
	public void testQREigenSmall() {
		MatrixBlock in = TestUtils.generateTestMatrixBlock(5, 5, 0.0, 1.0, 1.0, 5);
		testQREigen(in, 1e-4, 1);
	}

	@Test
	public void testQREigenComplexEVs() {
		MatrixBlock in = TestUtils.generateTestMatrixBlock(10, 10, 0.0, 1.0, 1.0, 2);
		testQREigen(in, 1e-4, 1);
	}

	@Test
	public void testQREigenSymLarge() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(50, 50, 0.0, 1.0, 1.0, 1);
		testQREigen(in, 1e-4, 1);
	}

	@Test
	public void testQREigenSymLargeMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(50, 50, 0.0, 1.0, 1.0, 1);
		testQREigen(in, 1e-4, threads);
	}

	private void testQREigen(MatrixBlock in, double tol, int threads) {
		try {
			long t1 = System.nanoTime();
			MatrixBlock[] m1 = LibCommonsMath.multiReturnOperations(in, "eigen", threads, 1);
			long t2 = System.nanoTime();
			MatrixBlock[] m2 = LibCommonsMath.multiReturnOperations(in, "eigen_qr", threads, 1);
			long t3 = System.nanoTime();

			LOG.error(
				"time eigen: " + (t2 - t1) + " time QR: " + (t3 - t2) + " QR speedup: " + ((double) (t2 - t1) / (t3 - t2)));
			TestUtils.compareMatrices(m1[0], m2[0], tol, "Result of eigenvalues of new eigendecomp function wrong");
			testEvecValues(m1[1], m2[1], tol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void testEvecValues(MatrixBlock a, MatrixBlock b, double tol) {
		double[][] m1 = DataConverter.convertToArray2DRowRealMatrix(a).getData();
		double[][] m2 = DataConverter.convertToArray2DRowRealMatrix(b).getData();

		for(int i = 0; i < m1.length; i++) {
			for(int j = 0; j < m1[0].length; j++) {
				if(!(TestUtils.compareCellValue(m1[i][j], m2[i][j], tol, false) ||
					TestUtils.compareCellValue(m1[i][j], -1 * m2[i][j], tol, false))) {
					Assert.fail("Result of eigenvectors of new eigendecomp function wrong " + m1[i][j] + "not equal "
						+ (-1 * m2[i][j]));
				}
			}
		}
	}
}
