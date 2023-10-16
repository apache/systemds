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
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MatrixMultiplyTest {
	protected static final Log LOG = LogFactory.getLog(MatrixMultiplyTest.class.getName());

	// left side
	private final MatrixBlock left;
	// right side
	private final MatrixBlock right;
	// expected result
	private final MatrixBlock exp;
	// parallelization degree
	private final int k;

	public MatrixMultiplyTest(int i, int j, int k, double s, double s2, int p) {
		try {
			this.left = TestUtils.ceil(TestUtils.generateTestMatrixBlock(i, j, -10, 10, i == 1 && j == 1 ? 1 : s, 13));
			this.right = TestUtils.ceil(TestUtils.generateTestMatrixBlock(j, k, -10, 10, k == 1 && k == 1 ? 1 : s2, 14));

			this.exp = multiply(left, right, 1);
			this.k = p;
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	@Parameters
	public static Collection<Object[]> data() {

		List<Object[]> tests = new ArrayList<>();
		try {
			double[] sparsities = new double[] {0.001, 0.1, 0.5};
			int[] is = new int[] {1, 3, 1024};
			int[] js = new int[] {1, 3, 1024};
			int[] ks = new int[] {1, 3, 1024};
			int[] par = new int[] {1, 4};

			for(int s = 0; s < sparsities.length; s++) {
				for(int s2 = 0; s2 < sparsities.length; s2++) {
					for(int p = 0; p < par.length; p++) {
						for(int i = 0; i < is.length; i++) {
							for(int j = 0; j < js.length; j++) {
								for(int k = 0; k < ks.length; k++) {
									tests.add(new Object[] {is[i], js[j], ks[k], sparsities[s], sparsities[s2], par[p]});
								}
							}
						}
					}
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	@Test
	public void testMultiplicationAsIs() {
		test(left, right);
	}

	@Test
	public void testLeftForceDense() {
		left.sparseToDense();
		test(left, right);
	}

	@Test
	public void testRightForceDense() {
		right.sparseToDense();
		test(left, right);
	}

	@Test
	public void testBothForceDense() {
		left.sparseToDense();
		right.sparseToDense();
		test(left, right);
	}

	@Test
	public void testLeftForceSparse() {
		left.denseToSparse(true);
		test(left, right);
	}

	@Test
	public void testRightForceSparse() {
		right.denseToSparse(true);
		test(left, right);
	}

	@Test
	public void testBothForceSparse() {
		left.denseToSparse(true);
		right.denseToSparse(true);
		test(left, right);
	}

	private void test(MatrixBlock a, MatrixBlock b) {
		try {
			MatrixBlock ret = multiply(a, b, k);

			boolean sparseLeft = a.isInSparseFormat();
			boolean sparseRight = b.isInSparseFormat();
			boolean sparseOut = exp.isInSparseFormat();
			String sparseErrMessage = "SparseLeft:" + sparseLeft + " SparseRight: " + sparseRight + " SparseOut:"
				+ sparseOut;
			String sizeErrMessage = size(a) + "  " + size(b) + "  " + size(exp);

			String totalMessage = "\n\n" + sizeErrMessage + "\n" + sparseErrMessage;

			if(ret.getNumRows() * ret.getNumColumns() < 1000 || ret.getNonZeros() < 100) {
				totalMessage += "\n\nExp" + exp;
				totalMessage += "\n\nAct" + ret;
			}

			assertEquals(totalMessage,exp.getNonZeros(), ret.getNonZeros());
			TestUtils.compareMatricesPercentageDistance(exp, ret, 0.999, 0.99999, totalMessage, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static String size(MatrixBlock a) {
		return a.getNumRows() + "x" + a.getNumColumns() + "n" + a.getNonZeros();
	}

	private static MatrixBlock multiply(MatrixBlock a, MatrixBlock b, int k) {
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator mult = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, k);
		return a.aggregateBinaryOperations(a, b, mult);
	}
}
