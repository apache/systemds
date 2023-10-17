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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MatrixToSparseOrDense {
	protected static final Log LOG = LogFactory.getLog(MatrixMultiplyTest.class.getName());

	private final MatrixBlock a;

	public MatrixToSparseOrDense(int rows, int cols, double sparsity) {

		this.a = TestUtils.generateTestMatrixBlock(rows, cols, -10, 10, sparsity, 42151);
	}

	@Parameters
	public static Collection<Object[]> data() {

		List<Object[]> tests = new ArrayList<>();
		try {
			double[] sparsities = new double[] {0.0001, 0.1, 0.5};
			int[] is = new int[] {1, 2, 10, 1302};
			int[] js = new int[] {1, 2, 5, 1203};

			for(int s = 0; s < sparsities.length; s++) {
				for(int i = 0; i < is.length; i++) {
					for(int j = 0; j < js.length; j++) {
						tests.add(new Object[] {is[i], js[j], sparsities[s]});

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
	public void testA() {
		MatrixBlock c = new MatrixBlock();
		c.copy(a);
		TestUtils.compareMatricesPercentageDistance(a, c, 1.0, 1.0, "");
	}

	@Test
	public void testB_MCSR() {
		MatrixBlock c = new MatrixBlock();
		c.copy(a);
		c.denseToSparse(false);
		TestUtils.compareMatricesPercentageDistance(a, c, 1.0, 1.0, "");
	}

	@Test
	public void testB_CSR() {
		MatrixBlock c = new MatrixBlock();
		c.copy(a);
		c.denseToSparse(true);
		TestUtils.compareMatricesPercentageDistance(a, c, 1.0, 1.0, "");
	}

	@Test
	public void testCDense() {
		MatrixBlock c = new MatrixBlock();
		c.copy(a);
		c.sparseToDense();
		TestUtils.compareMatricesPercentageDistance(a, c, 1.0, 1.0, "");
	}
}
