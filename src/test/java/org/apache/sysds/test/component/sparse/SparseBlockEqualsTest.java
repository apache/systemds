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

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;

@RunWith(Enclosed.class)
public class SparseBlockEqualsTest {

	@RunWith(Parameterized.class)
	public static class SparseBlockEqualsSparseBlockTest extends AutomatedTestBase {

		@Override
		public void setUp() {
			TestUtils.clearAssertionInformation();
		}

		private final SparseBlock.Type type1;
		private final SparseBlock.Type type2;

		public SparseBlockEqualsSparseBlockTest(SparseBlock.Type type1, SparseBlock.Type type2) {
			this.type1 = type1;
			this.type2 = type2;
		}

		@Parameterized.Parameters(name = "{0} vs {1}")
		public static Iterable<Object[]> types() {
			SparseBlock.Type[] types = SparseBlock.Type.values();
			ArrayList<Object[]> params = new ArrayList<>();

			for (int i = 0; i < types.length; i++) {
				for (int j = i; j < types.length; j++) {
					params.add(new Object[]{types[i], types[j]});
				}
			}

			return params;
		}

		@Test
		public void testSparseBlockEquals() {
			runSparseBlockEqualsTest(type1, type2);
		}

		@Test
		public void testSparseBlockNotEqualsColIdx() {
			runSparseBlockNotEqualsColIdxTest(type1, type2);
		}

		@Test
		public void testSparseBlockNotEqualsEmptyRow() {
			runSparseBlockNotEqualsEmptyRowTest(type1, type2);
		}
	}

	@RunWith(Parameterized.class)
	public static class SparseBlockEqualsDenseValuesTest extends AutomatedTestBase {

		@Override
		public void setUp() {
			TestUtils.clearAssertionInformation();
		}

		private final SparseBlock.Type type;

		public SparseBlockEqualsDenseValuesTest(SparseBlock.Type type) {
			this.type = type;
		}

		@Parameterized.Parameters(name = "{0}")
		public static Iterable<Object[]> types() {
			ArrayList<Object[]> params = new ArrayList<>();
			for (SparseBlock.Type t : SparseBlock.Type.values()) {
				params.add(new Object[]{t});
			}
			return params;
		}

		@Test
		public void testSparseBlockNotEqualsNonSparseBlock() {
			runSparseBlockNotEqualsNonSparseBlockTest(type);
		}

		@Test
		public void testSparseBlockNotEqualsDenseValuesEmptyRow() {
			runSparseBlockNotEqualsDenseValuesEmptyRowTest(type);
		}

		@Test
		public void testSparseBlockNotEqualsDenseValuesNonZero() {
			runSparseBlockNotEqualsDenseValuesNonZeroTest(type);
		}

		@Test
		public void testSparseBlockNotEqualsDenseValuesAdditionalNonZero() {
			runSparseBlockNotEqualsDenseValuesAdditionalNonZeroTest(type);
		}
	}

	private static void runSparseBlockEqualsTest(SparseBlock.Type type1, SparseBlock.Type type2) {
		double[][] A = new double[][]{{1., 2., 3.}, {0., 0., 0.}, {0., 4., 0.}, {0., 0., 5.}, {6., 0., 0.}, {0., 0., 7.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock1 = SparseBlockFactory.copySparseBlock(type1, srtmp, true);
		SparseBlock sblock2 = SparseBlockFactory.copySparseBlock(type2, srtmp, true);

		assertEquals(sblock1, sblock2);
	}

	private static void runSparseBlockNotEqualsColIdxTest(SparseBlock.Type type1, SparseBlock.Type type2) {
		double[][] A = new double[][]{{1., 2., 3.}, {0., 0., 0.}, {0., 4., 0.}, {0., 0., 5.}, {6., 0., 0.}, {0., 0., 7.}};
		double[][] B = new double[][]{{1., 2., 3.}, {0., 0., 0.}, {0., 0., 4.}, {0., 0., 5.}, {6., 0., 0.}, {0., 0., 7.}};

		MatrixBlock mbtmp1 = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp1 = mbtmp1.getSparseBlock();
		SparseBlock sblock1 = SparseBlockFactory.copySparseBlock(type1, srtmp1, true);

		MatrixBlock mbtmp2 = DataConverter.convertToMatrixBlock(B);
		SparseBlock srtmp2 = mbtmp2.getSparseBlock();
		SparseBlock sblock2 = SparseBlockFactory.copySparseBlock(type2, srtmp2, true);

		assertNotEquals("should not be equal: " + type1 + " " + type2, sblock1, sblock2);
	}

	private static void runSparseBlockNotEqualsEmptyRowTest(SparseBlock.Type type1, SparseBlock.Type type2) {
		double[][] A = new double[][]{{1., 2., 3.}, {0., 0., 0.}, {0., 4., 0.}, {0., 0., 5.}, {6., 0., 0.}, {0., 0., 7.}};
		double[][] B = new double[][]{{1., 2., 3.}, {0., 4., 0.}, {0., 0., 0.}, {0., 0., 5.}, {6., 0., 0.}, {0., 0., 7.}};

		MatrixBlock mbtmp1 = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp1 = mbtmp1.getSparseBlock();
		SparseBlock sblock1 = SparseBlockFactory.copySparseBlock(type1, srtmp1, true);

		MatrixBlock mbtmp2 = DataConverter.convertToMatrixBlock(B);
		SparseBlock srtmp2 = mbtmp2.getSparseBlock();
		SparseBlock sblock2 = SparseBlockFactory.copySparseBlock(type2, srtmp2, true);

		assertNotEquals("should not be equal: " + type1 + " " + type2, sblock1, sblock2);
	}

	private static void runSparseBlockNotEqualsNonSparseBlockTest(SparseBlock.Type type) {
		double[][] A = new double[][]{{1., 0., 3.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(type, srtmp, true);

		SparseRowVector srv = new SparseRowVector(A[0], new int[]{0, 1, 2});

		assertNotEquals("should not be equal: " + type, sblock, srv);
	}

	private static void runSparseBlockNotEqualsDenseValuesEmptyRowTest(SparseBlock.Type type) {
		double[][] A = new double[][]{{1., 0., 3.}, {0., 0., 0.}, {0., 0., 0.}, {4., 0., 6.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(type, srtmp, true);

		double[] denseValues = new double[]{1., 0., 3., 0., 0., 0., 1., 1., 1., 4., 0., 6.};

		assertFalse("should not be equal: " + type, sblock.equals(denseValues, 3, 1e-10));
	}

	private static void runSparseBlockNotEqualsDenseValuesNonZeroTest(SparseBlock.Type type) {
		double[][] A = new double[][]{{1., 0., 3.}, {0., 0., 0.}, {0., 0., 0.},{0., 0., 1.}, {4., 0., 6.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(type, srtmp, true);

		double[] denseValues = new double[]{1., 0., 3., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 0., 6.};

		assertFalse("should not be equal: " + type, sblock.equals(denseValues, 3, 1e-10));
	}

	private static void runSparseBlockNotEqualsDenseValuesAdditionalNonZeroTest(SparseBlock.Type type) {
		double[][] A = new double[][]{{1., 0., 3.}, {0., 0., 0.}, {0., 0., 0.}, {4., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(type, srtmp, true);

		double[] denseValues = new double[]{1., 0., 3., 0., 0., 0., 0., 0., 0., 4., 0., 6.};

		assertFalse("should not be equal: " + type, sblock.equals(denseValues, 3, 1e-10));
	}
}
