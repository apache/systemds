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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

public class EigenDecompTest {

	protected static final Log LOG = LogFactory.getLog(EigenDecompTest.class.getName());

	private enum type {
		COMMONS, LANCZOS, QR,
	}

	@Test
	public void testLanczosSimple() {
		MatrixBlock in = new MatrixBlock(4, 4, new double[] {4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1});
		testEigen(in, 1e-4, 1, type.LANCZOS);
	}

	@Test
	public void testLanczosSmall() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, 1, type.LANCZOS);
	}

	@Test
	public void testLanczosMedium() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(12, 12, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, 1, type.LANCZOS);
	}

	@Test
	@Ignore
	public void testLanczosLarge() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(100, 100, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, 1, type.LANCZOS);
	}

	@Test
	@Ignore
	public void testLanczosLargeMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(100, 100, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, threads, type.LANCZOS);
	}

	@Test
	public void testQREigenSimple() {
		MatrixBlock in = new MatrixBlock(4, 4,
			new double[] {52, 30, 49, 28, 30, 50, 8, 44, 49, 8, 46, 16, 28, 44, 16, 22});
		testEigen(in, 1e-4, 1, type.QR);
	}

	@Test
	public void testQREigenSymSmall() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-3, 1, type.QR);
	}

	@Test
	public void testQREigenSymSmallMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(10, 10, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-3, threads, type.QR);
	}

	@Test
	@Ignore
	public void testQREigenSymLarge() {
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(50, 50, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, 1, type.QR);
	}

	@Test
	@Ignore
	public void testQREigenSymLargeMT() {
		int threads = 10;
		MatrixBlock in = TestUtils.generateTestMatrixBlockSym(50, 50, 0.0, 1.0, 1.0, 1);
		testEigen(in, 1e-4, threads, type.QR);
	}

	private void testEigen(MatrixBlock in, double tol, int threads, type t) {
		try {
			MatrixBlock[] m = null;
			switch(t) {
				case COMMONS:
					m = LibCommonsMath.multiReturnOperations(in, "eigen", threads, 1);
					break;
				case LANCZOS:
					m = LibCommonsMath.multiReturnOperations(in, "eigen_lanczos", threads, 1);
					break;
				case QR:
					m = LibCommonsMath.multiReturnOperations(in, "eigen_qr", threads, 1);
					break;
				default:
					throw new NotImplementedException();
			}

			isValidDecomposition(in, m[1], m[0], tol, t.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void isValidDecomposition(MatrixBlock A, MatrixBlock V, MatrixBlock vD, double tol, String message) {
		// Any eigen decomposition is valid if A = V %*% D %*% t(V)
		// A is the input of the eigen decomposition
		// D is the eigen values in a diagonal matrix
		// V is the eigen vectors

		final int m = V.getNumColumns();
		final MatrixBlock D = new MatrixBlock(m, m, false);
		LibMatrixReorg.diag(vD, D);

		MatrixBlock VD = new MatrixBlock();
		LibMatrixMult.matrixMult(V, D, VD);

		MatrixBlock VDtV = new MatrixBlock();
		LibMatrixMult.matrixMult(VD, LibMatrixReorg.transpose(V), VDtV);

		TestUtils.compareMatrices(A, VDtV, tol, message);
	}
}
