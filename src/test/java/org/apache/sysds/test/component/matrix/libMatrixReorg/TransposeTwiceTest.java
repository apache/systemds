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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class TransposeTwiceTest {
	protected static final Log LOG = LogFactory.getLog(TransposeTwiceTest.class.getName());

	long old = LibMatrixReorg.PAR_NUMCELL_THRESHOLD;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			double[] sparsities = new double[] {0.0, 0.001, 0.1, 0.2, 0.35, 0.5, 1.0};
			int[] rowsCols = new int[] {1, 2, 4, 5, 10, 30, 50, 100, 300};
			int[] parallelization = new int[] {-1, 1, 5};

			// smaller cases
			for(int rows : rowsCols) {
				for(int cols : rowsCols) {
					for(double sp : sparsities) {
						MatrixBlock mb = TestUtils.generateTestMatrixBlock(rows, cols, -10, 10, sp, 42);
						mb.examSparsity(false);
						for(int k1 : parallelization) {
							for(int k2 : parallelization) {
								tests.add(new Object[] {mb, k1, k2});
							}
						}
					}
				}
			}

			// Larger edge case
			MatrixBlock mb;
			rowsCols = new int[] {1, 2, 5, 10};
			for(int rows : rowsCols) {
				for(double sp : sparsities) {
					if(sp > 0.4)
						continue;
					mb = TestUtils.generateTestMatrixBlock(rows, 5000, -10, 10, sp, 43);
					tests.add(new Object[] {mb, 16, 16});
				}
			}
		}
		catch(Exception e) {
			fail(e.getMessage());
		}
		return tests;
	}

	@Parameterized.Parameter
	public MatrixBlock in;
	@Parameterized.Parameter(1)
	public int k1;
	@Parameterized.Parameter(2)
	public int k2;

	@Test
	public void testTransposeBackAndForth() {
		try {
			LibMatrixReorg.PAR_NUMCELL_THRESHOLD = 100;

			MatrixBlock inT = LibMatrixReorg.transpose(in, k1);
			MatrixBlock inTT = LibMatrixReorg.transpose(inT, k2);

			TestUtils.compareMatricesBitAvgDistance(in, inTT, 0, 0, "Not equal transpose result");
			assertEquals(in.getNonZeros(), inT.getNonZeros());
			assertEquals(in.getNonZeros(), inTT.getNonZeros());
			assertEquals(in.getNumRows(), inT.getNumColumns());
			assertEquals(in.getNumColumns(), inT.getNumRows());
			compareTransposed(in, inT);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed in transposition");
		}
		finally {

			LibMatrixReorg.PAR_NUMCELL_THRESHOLD = old;
		}
	}

	private void compareTransposed(MatrixBlock in, MatrixBlock inT) {
		int nRow = in.getNumRows();
		int nCol = in.getNumColumns();
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				assertEquals(in.get(i, j), inT.get(j, i), 0.0);
			}
		}
	}
}
