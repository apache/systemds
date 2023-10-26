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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class TSMMTest {

	protected static final Log LOG = LogFactory.getLog(TSMMTest.class.getName());

	@Parameterized.Parameter
	public MatrixBlock in;
	@Parameterized.Parameter(1)
	public int k;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		MatrixBlock mb;
		final double[] spar = new double[] {0.3, 0.1, 0.01};
		final int[] cols = new int[] {10, 6, 4, 3, 2, 1};
		final int[] threads = new int[] {1, 3, 10, 11, 16, 32, 44};
		final int[] rows = new int[] {10};
		for(int i = 0; i < 3; i++) { // seeds
			for(int s = 0; s < spar.length; s++) {
				for(int c = 0; c < cols.length; c++) {
					for(int r = 0; r < rows.length; r++) {
						mb = TestUtils.round(TestUtils.generateTestMatrixBlock(rows[r], cols[c], 1, 10, spar[s], i));
						for(int t = 0; t < threads.length; t++)
							tests.add(new Object[] {mb, threads[t]});
					}
				}

			}
		}
		MatrixBlock mb1 = new MatrixBlock(100, 100, true); // dense cols sparse block
		for(int j : new int[] {1, 4, 44, 87})
			for(int i = 0; i < 100; i++)
				mb1.quickSetValue(i, j, 100);
		for(int t = 0; t < threads.length; t++)
			tests.add(new Object[] {mb1, threads[t]});

		MatrixBlock mb2 = new MatrixBlock(100, 100, true); // sparse but specific
		for(int j : new int[] {44, 87})
			for(int i : new int[] {56, 92})
				mb2.quickSetValue(i, j, 100);
		for(int t = 0; t < threads.length; t++)
			tests.add(new Object[] {mb2, threads[t]});

		MatrixBlock mb3 = new MatrixBlock(100, 100, true); // dense rows sparse block
		for(int j : new int[] {1, 4, 44, 87})
			for(int i = 0; i < 100; i++)
				mb3.quickSetValue(j, i, 100);
		for(int t = 0; t < threads.length; t++)
			tests.add(new Object[] {mb3, threads[t]});
		return tests;
	}

	@Test
	public void testTSMMLeftSparseVSDense() {
		final MMTSJType mType = MMTSJType.LEFT;
		final MatrixBlock expected = in.transposeSelfMatrixMultOperations(null, mType, 1);

		if(k > 1) // test multi-threaded
			testCompare(expected, in, "Compare single vs multithread");

		final boolean isSparse = in.isInSparseFormat();

		if(isSparse) {
			MatrixBlock m2 = new MatrixBlock();
			m2.copy(in);
			m2.sparseToDense();
			testCompare(expected, m2, "Compare sparse vs dense");
		}
		else {
			MatrixBlock m2 = new MatrixBlock();
			m2.copy(in);
			m2.denseToSparse(true);
			testCompare(expected, m2, "Compare dense vs CSR Sparse");

			MatrixBlock m3 = new MatrixBlock();
			m3.copy(in);
			m3.denseToSparse(false);
			testCompare(expected, m3, "Compare dense vs MCSR Sparse");
		}
	}

	private void testCompare(MatrixBlock expected, MatrixBlock m2, String message) {
		final MMTSJType mType = MMTSJType.LEFT;
		final MatrixBlock actual = m2.transposeSelfMatrixMultOperations(null, mType, k);
		// final String inString = m2.toString();
		TestUtils.compareMatricesBitAvgDistance(expected, actual, 10L, 256L, message);
	}
}
