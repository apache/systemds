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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.lib.CLALibSelectionMult;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.mapping.MappingTestUtil;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CLALibSelectionMultTest {
	protected static final Log LOG = LogFactory.getLog(CombineGroupsTest.class.getName());

	@Parameterized.Parameter(0)
	public String s;
	@Parameterized.Parameter(1)
	public MatrixBlock mb;
	@Parameterized.Parameter(2)
	public CompressedMatrixBlock cmb;
	@Parameterized.Parameter(3)
	public MatrixBlock mb2;

	@BeforeClass
	public static void setup() {
		Thread.currentThread().setName("main_test_" + Thread.currentThread().getId());
	}

	@Parameters(name = "{0}")
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		MatrixBlock mb;
		CompressedMatrixBlock cmb;
		List<AColGroup> gs;

		try {
			mb = TestUtils.generateTestMatrixBlock(200, 50, -10, 10, 1.0, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "Normal");

			mb = TestUtils.generateTestMatrixBlock(1020, 50, -10, 10, 1.0, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "NormalLarge");

			cmb = (CompressedMatrixBlock) cmb.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 2), null);
			mb = cmb.decompress();
			genTests(tests, mb, cmb, "NormalP2");

			cmb = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
			cmb.allocateColGroup(ColGroupUncompressed.create(mb));
			genTests(tests, mb, cmb, "UncompressAbleGroup");

			mb = TestUtils.generateTestMatrixBlock(200, 10, -5, 5, 1.0, 32);
			mb = TestUtils.round(mb);
			gs = new ArrayList<>();
			gs.add(ColGroupUncompressed.create(mb));
			gs.add(ColGroupUncompressed.create(mb));
			cmb = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), 1000, true, gs);
			mb = cmb.decompress();
			genTests(tests, mb, cmb, "UncompressAbleGroup2");

			MatrixBlock mb2 = new MatrixBlock(10, 10, 132.0);
			cmb = CompressedMatrixBlockFactory.createConstant(10, 10, 132.0);
			genTests(tests, mb2, cmb, "Const");

			gs = new ArrayList<>();
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), 100.0));
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), 32.0));
			cmb = new CompressedMatrixBlock(10, 10, 100, true, gs);
			genTests(tests, cmb.getUncompressed(), cmb, "OverlappingConst");

			gs = new ArrayList<>();
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), new double[] {13.0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), new double[] {13.0, 32., 0, 0, 0, 0, 0, 0, 0, 0}));
			cmb = new CompressedMatrixBlock(10, 10, 100, true, gs);
			genTests(tests, cmb.getUncompressed(), cmb, "OverlappingSparseConst");

			gs = new ArrayList<>();
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), new double[] {13.0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
			IDictionary id = IdentityDictionary.create(10);
			AMapToData d = MappingTestUtil.createRandomMap(100, 10, new Random(23));
			AColGroup idg = ColGroupDDC.create(ColIndexFactory.create(10, 100), id, d, null);
			gs.add(idg);
			cmb = new CompressedMatrixBlock(100, 100, 100, true, gs);
			genTests(tests, cmb.getUncompressed(), cmb, "OverlappingSparseConst");

			mb = TestUtils.generateTestMatrixBlock(200, 16, -10, 10, 0.04, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "Sparse");

			mb = TestUtils.generateTestMatrixBlock(200, 16, -10, 10, 0.2, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "Sparse2");

			mb = TestUtils.generateTestMatrixBlock(1023, 16, -10, 10, 0.01, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "SparseLarge");

			d = MappingTestUtil.createRandomMap(100, 10, new Random(23));
			idg = ColGroupDDC.create(ColIndexFactory.create(10), id, d, null);
			cmb = new CompressedMatrixBlock(100, 10);
			cmb.allocateColGroup(idg);
			mb = CompressedMatrixBlock.getUncompressed(cmb);
			genTests(tests, mb, cmb, "Identity");

			d = MappingTestUtil.createRandomMap(100, 10, new Random(23));
			idg = ColGroupDDC.create(ColIndexFactory.createI(0,1,2,3,4,6,7,8,9,10), id, d, null);
			cmb = new CompressedMatrixBlock(100, 11);
			cmb.allocateColGroup(idg);
			mb = CompressedMatrixBlock.getUncompressed(cmb);
			genTests(tests, mb, cmb, "Identity2");

			id = IdentityDictionary.create(10, true);

			d = MappingTestUtil.createRandomMap(100, 11, new Random(33));
			idg = ColGroupDDC.create(ColIndexFactory.createI(0,1,2,3,4,6,7,8,9,10), id, d, null);
			cmb = new CompressedMatrixBlock(100, 11);
			cmb.allocateColGroup(idg);
			mb = CompressedMatrixBlock.getUncompressed(cmb);
			genTests(tests, mb, cmb, "Identity_empty");

			AColGroup empty = new ColGroupEmpty(ColIndexFactory.create(10));
			cmb = new CompressedMatrixBlock(100, 10);
			cmb.allocateColGroup(empty);
			genTests(tests, new MatrixBlock(10, 10, true), cmb, "Empty");

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static void genTests(List<Object[]> tests, MatrixBlock mb, MatrixBlock cmb, String version) {

		MatrixBlock tmp;
		MatrixBlock tcmb;

		final int nRow = cmb.getNumRows();
		// final int nCol = cmb.getNumColumns();

		tcmb = createSelectionMatrix(nRow, 30, false);
		// decompressed for uncompressed operation
		tmp = CompressedMatrixBlock.getUncompressed(tcmb);
		// compress the transposed version of it.
		tcmb = CompressedMatrixBlockFactory.compress(tmp.transpose(), 1).getLeft();
		tests.add(new Object[] {version + "_selection", mb, cmb, tmp});

		tcmb = createSelectionMatrix(nRow, 30, true);
		// decompressed for uncompressed operation
		tmp = CompressedMatrixBlock.getUncompressed(tcmb);
		// compress the transposed version of it.
		tcmb = CompressedMatrixBlockFactory.compress(tmp.transpose(), 1).getLeft();
		tests.add(new Object[] {version + "_selection_with_empty", mb, cmb, tmp});

		tcmb = createSelectionMatrix(nRow, 1002, true);
		// decompressed for uncompressed operation
		tmp = CompressedMatrixBlock.getUncompressed(tcmb);
		// compress the transposed version of it.
		tcmb = CompressedMatrixBlockFactory.compress(tmp.transpose(), 1).getLeft();
		tests.add(new Object[] {version + "_selection_with_empty_1002", mb, cmb, tmp});

	}

	public static MatrixBlock createSelectionMatrix(final int nRow, final int nRowLeft, boolean emptyRows) {
		MatrixBlock tcmb;
		IDictionary id = IdentityDictionary.create(nRow, emptyRows);
		AMapToData d = MappingTestUtil.createRandomMap(nRowLeft, nRow + (emptyRows ? 1 : 0), new Random(33));
		AColGroup idg = ColGroupDDC.create(ColIndexFactory.create(nRow), id, d, null);
		tcmb = new CompressedMatrixBlock(nRowLeft, nRow);
		((CompressedMatrixBlock) tcmb).allocateColGroup(idg);
		return tcmb;
	}

	@Test
	public void testMultiplicationSingleThread() {
		try {
			exec(mb, cmb, mb2, 1);
		}
		catch(Exception e) {
			if(!causeWasNotImplemented(e)) {
				e.printStackTrace();
				fail(e.getMessage());
			}
		}
	}

	@Test
	public void testMultiplicationUnknownNonZeros() {
		try {
			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getNonZeros()).thenReturn(-1L);
			exec(mb, spy, mb2, 1);
		}
		catch(Exception e) {
			if(!causeWasNotImplemented(e)) {
				e.printStackTrace();
				fail(e.getMessage());
			}
		}
	}

	@Test
	public void testMultiplicationParallel() {
		try {
			exec(mb, cmb, mb2, 4);
		}
		catch(Exception e) {
			if(!causeWasNotImplemented(e)) {
				e.printStackTrace();
				fail(e.getMessage());
			}
		}
	}

	@Test
	public void testMultiplicationParallelAllocatedRet() {
		try {
			execR(mb, cmb, mb2, new MatrixBlock(), 4);
		}
		catch(Exception e) {
			if(!causeWasNotImplemented(e)) {
				e.printStackTrace();
				fail(e.getMessage());
			}
		}
	}

	private boolean causeWasNotImplemented(Throwable e) {
		return e instanceof NotImplementedException || //
			(e.getCause() != null && causeWasNotImplemented(e.getCause()));
	}

	private static void exec(MatrixBlock mb1, CompressedMatrixBlock cmb1, MatrixBlock mb2, int k) {
		execR(mb1, cmb1, mb2, null, k);
	}

	private static void execR(MatrixBlock mb1, CompressedMatrixBlock cmb1, MatrixBlock mb2, MatrixBlock ret, int k) {
		MatrixBlock cRet = CLALibSelectionMult.leftSelection(cmb1, mb2, ret, k);
		MatrixBlock uRet = LibMatrixMult.matrixMult(CompressedMatrixBlock.getUncompressed(mb2), mb1, k);
		compare(cRet, uRet);
	}

	private static void compare(MatrixBlock cRet, MatrixBlock uRet) {
		TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0);
	}

}
