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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class SparseBlockContainsTest extends AutomatedTestBase
{
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockContainsNoMatchCOO()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockContainsNoMatchCSC()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockContainsNoMatchCSR()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockContainsNoMatchDCSR()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockContainsNoMatchMCSC()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockContainsNoMatchMCSR()  {
		runSparseBlockContainsNoMatchTest(SparseBlock.Type.MCSR);
	}

	@Test
	public void testSparseBlockContainsNaNCOO()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockContainsNaNCSC()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockContainsNaNCSR()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockContainsNaNDCSR()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockContainsNaNMCSC()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockContainsNaNMCSR()  {
		runSparseBlockContainsNaNTest(SparseBlock.Type.MCSR);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortCOO()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortCSC()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortCSR()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortDCSR()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortMCSC()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockContainsEarlyAbortMCSR()  {
		runSparseBlockContainsEarlyAbortTest(SparseBlock.Type.MCSR);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsCOO()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsCSC()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsCSR()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsDCSR()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsMCSC()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockContainsPatternLongerThanRowsMCSR()  {
		runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type.MCSR);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroCOO()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroCSC()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroCSR()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroDCSR()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroMCSC()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.MCSC);
	}

	@Test
	public void testSparseBlockContainsPatternContainsZeroMCSR()  {
		runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type.MCSR);
	}

	private void runSparseBlockContainsNoMatchTest(SparseBlock.Type btype) {
		double[] pattern = new double[]{1., 2., 3.};
		double[][] A = new double[][]{{4., 5., 6.}, {7., 8., 9.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		List<Integer> result = sblock.contains(pattern, false);
		assertEquals(List.of(), result);
	}

	private void runSparseBlockContainsNaNTest(SparseBlock.Type btype) {
		double[] pattern = new double[]{Double.NaN, 2., 3.};
		double[][] A = new double[][]{{Double.NaN, 2., 3.}, {1., 2., 3.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		List<Integer> result = sblock.contains(pattern, false);
		assertEquals(List.of(0), result);
	}

	private void runSparseBlockContainsEarlyAbortTest(SparseBlock.Type btype) {
		double[] pattern = new double[]{1., 2., 3.};
		double[][] A = new double[][]{{0., 0., 0.}, {1., 2., 3.}, {1., 2., 3.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		List<Integer> result = sblock.contains(pattern, true);
		assertEquals(List.of(1), result);
	}

	private void runSparseBlockContainsPatternLongerThanRowsTest(SparseBlock.Type btype) {
		double[] pattern = new double[]{1., 2., 3., 4.};
		double[][] A = new double[][]{{0., 0., 0.}, {1., 2., 3.}, {1., 2., 3.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		List<Integer> result = sblock.contains(pattern, false);
		assertEquals(List.of(), result);
	}

	private void runSparseBlockContainsPatternContainsZeroTest(SparseBlock.Type btype) {
		double[] pattern = new double[]{0., 1., 2.};
		double[][] A = new double[][]{{0., 1., 2.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}, {0., 1., 2.},  {1., 2., 0.}};

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		List<Integer> result = sblock.contains(pattern, false);
		assertEquals(List.of(0, 4), result);
	}
}
