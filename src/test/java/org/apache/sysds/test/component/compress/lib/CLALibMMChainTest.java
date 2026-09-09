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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibTSMM;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.mapping.MappingTestUtil;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Targeted tests for the compressed transpose-self multiply ({@link CLALibTSMM}) and the XtXv mm-chain fast path that
 * was added in {@code CLALibMMChain}. The fast path triggers when the input has fewer than five column groups and more
 * than thirty columns, in which case the chain is computed as {@code (t(X) %*% X) %*% v}.
 */
public class CLALibMMChainTest {
	protected static final Log LOG = LogFactory.getLog(CLALibMMChainTest.class.getName());

	@BeforeClass
	public static void setup() {
		Thread.currentThread().setName("main_test_" + Thread.currentThread().getId());
	}

	/**
	 * Build a compressed matrix backed by a single DDC column group spanning all {@code nCol} columns. This guarantees a
	 * single (non-uncompressed) column group, which is what triggers the mm-chain fast path for wide enough matrices.
	 */
	private static CompressedMatrixBlock singleDDC(int nRow, int nCol, int nVal, int seed) {
		Random r = new Random(seed);
		double[] dictValues = new double[nVal * nCol];
		for(int i = 0; i < dictValues.length; i++)
			dictValues[i] = Math.round(r.nextDouble() * 20 - 10);
		IDictionary dict = Dictionary.create(dictValues);
		AMapToData data = MappingTestUtil.createRandomMap(nRow, nVal, r);
		AColGroup g = ColGroupDDC.create(ColIndexFactory.create(nCol), dict, data, null);
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(nRow, nCol);
		cmb.allocateColGroup(g);
		cmb.recomputeNonZeros();
		return cmb;
	}

	private static CompressedMatrixBlock uncompressedGroup(int nRow, int nCol, int seed) {
		MatrixBlock mb = TestUtils.round(TestUtils.generateTestMatrixBlock(nRow, nCol, -10, 10, 1.0, seed));
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(nRow, nCol);
		cmb.allocateColGroup(ColGroupUncompressed.create(mb));
		cmb.recomputeNonZeros();
		return cmb;
	}

	private static CompressedMatrixBlock empty(int nRow, int nCol) {
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(nRow, nCol);
		cmb.allocateColGroup(new ColGroupEmpty(ColIndexFactory.create(nCol)));
		cmb.recomputeNonZeros();
		return cmb;
	}

	@Test
	public void tsmmWideSingleThread() {
		execTSMM(singleDDC(200, 40, 6, 1), 1);
	}

	@Test
	public void tsmmWideParallel() {
		execTSMM(singleDDC(200, 40, 6, 2), 4);
	}

	@Test
	public void tsmmNarrowSingleThread() {
		execTSMM(singleDDC(200, 8, 4, 3), 1);
	}

	@Test
	public void tsmmNarrowParallel() {
		execTSMM(singleDDC(200, 8, 4, 4), 4);
	}

	@Test
	public void tsmmUncompressedGroupSingleThread() {
		// A compressed block holding an uncompressed column group must fall back to the dense tsmm path.
		execTSMM(uncompressedGroup(150, 12, 5), 1);
	}

	@Test
	public void tsmmUncompressedGroupParallel() {
		execTSMM(uncompressedGroup(150, 12, 6), 4);
	}

	@Test
	public void tsmmEmpty() {
		CompressedMatrixBlock cmb = empty(100, 13);
		MatrixBlock ret = CLALibTSMM.leftMultByTransposeSelf(cmb, 1);
		assertEquals(13, ret.getNumRows());
		assertEquals(13, ret.getNumColumns());
		assertTrue("empty input must produce an empty result", ret.isEmptyBlock(false));
	}

	@Test
	public void tsmmRetReused() {
		// A non-null ret must be reset and reused, producing the same result as a fresh allocation.
		CompressedMatrixBlock cmb = singleDDC(120, 36, 5, 7);
		MatrixBlock preAllocated = new MatrixBlock(3, 3, 99.0);
		preAllocated.allocateDenseBlock();
		MatrixBlock cRet = CLALibTSMM.leftMultByTransposeSelf(cmb, preAllocated, 4);
		MatrixBlock uRet = CompressedMatrixBlock.getUncompressed(cmb)
			.transposeSelfMatrixMultOperations(new MatrixBlock(), MMTSJType.LEFT, 4);
		TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0);
	}

	@Test
	public void tsmmRetNull() {
		// Explicitly exercise the null-ret allocation branch of the helper.
		CompressedMatrixBlock cmb = singleDDC(120, 36, 5, 8);
		MatrixBlock cRet = CLALibTSMM.leftMultByTransposeSelf(cmb, null, 1);
		MatrixBlock uRet = CompressedMatrixBlock.getUncompressed(cmb)
			.transposeSelfMatrixMultOperations(new MatrixBlock(), MMTSJType.LEFT, 1);
		TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0);
	}

	private static void execTSMM(CompressedMatrixBlock cmb, int k) {
		try {
			MatrixBlock cRet = CLALibTSMM.leftMultByTransposeSelf(cmb, k);
			MatrixBlock uRet = CompressedMatrixBlock.getUncompressed(cmb)
				.transposeSelfMatrixMultOperations(new MatrixBlock(), MMTSJType.LEFT, k);
			assertEquals(cmb.getNumColumns(), cRet.getNumRows());
			assertEquals(cmb.getNumColumns(), cRet.getNumColumns());
			TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void mmChainFastPathSingleThread() {
		// 40 columns, single column group -> XtXv fast path.
		execMMChain(singleDDC(200, 40, 6, 11), 1);
	}

	@Test
	public void mmChainFastPathParallel() {
		execMMChain(singleDDC(200, 40, 6, 12), 4);
	}

	@Test
	public void mmChainFastPathFewGroups() {
		// Two column groups (< 5) over 40 columns still triggers the fast path.
		execMMChain(twoGroups(200, 40, 13), 4);
	}

	@Test
	public void mmChainRegularPathNarrow() {
		// Only 20 columns -> below the width threshold, exercises the regular (non fast) chain path.
		execMMChain(singleDDC(200, 20, 6, 14), 4);
	}

	private static CompressedMatrixBlock twoGroups(int nRow, int nCol, int seed) {
		final int half = nCol / 2;
		Random r = new Random(seed);
		List<AColGroup> gs = new ArrayList<>();
		gs.add(ddcGroup(nRow, ColIndexFactory.create(0, half), 5, r));
		gs.add(ddcGroup(nRow, ColIndexFactory.create(half, nCol), 5, r));
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(nRow, nCol);
		cmb.allocateColGroupList(gs);
		cmb.recomputeNonZeros();
		return cmb;
	}

	private static AColGroup ddcGroup(int nRow, org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex cols,
		int nVal, Random r) {
		int nCol = cols.size();
		double[] dictValues = new double[nVal * nCol];
		for(int i = 0; i < dictValues.length; i++)
			dictValues[i] = Math.round(r.nextDouble() * 20 - 10);
		IDictionary dict = Dictionary.create(dictValues);
		AMapToData data = MappingTestUtil.createRandomMap(nRow, nVal, r);
		return ColGroupDDC.create(cols, dict, data, null);
	}

	@Test
	public void mmChainWideRecompressedDDC() {
		// Mirrors the e2e CompressedTestBase#testMatrixMultChainXtXvWide flow: tile a narrow matrix until it
		// exceeds the 30-column fast-path threshold, recompress it, then validate XtXv against uncompressed.
		execMMChainWide(TestUtils.round(TestUtils.generateTestMatrixBlock(300, 4, -10, 10, 1.0, 21)), 1);
	}

	@Test
	public void mmChainWideRecompressedSparse() {
		execMMChainWide(TestUtils.round(TestUtils.generateTestMatrixBlock(300, 3, 1, 5, 0.2, 22)), 4);
	}

	private static void execMMChainWide(MatrixBlock base, int k) {
		try {
			final int nCol = base.getNumColumns();
			final int reps = (int) Math.ceil(31.0 / nCol) + 1;
			MatrixBlock wide = base;
			for(int i = 1; i < reps; i++)
				wide = wide.append(base, new MatrixBlock(), true);
			assertTrue("widened matrix must exceed the fast-path threshold", wide.getNumColumns() > 30);

			MatrixBlock wideC = CompressedMatrixBlockFactory.compress(wide, k).getLeft();
			assertTrue("tiled matrix should compress", wideC instanceof CompressedMatrixBlock);

			MatrixBlock v = TestUtils.generateTestMatrixBlock(wide.getNumColumns(), 1, 0.9, 1.5, 1.0, 3);
			MatrixBlock uRet = wide.chainMatrixMultOperations(v, null, new MatrixBlock(), ChainType.XtXv, k);
			MatrixBlock cRet = wideC.chainMatrixMultOperations(v, null, new MatrixBlock(), ChainType.XtXv, k);
			TestUtils.compareMatrices(uRet, cRet, 1e-6, "wide recompressed mm-chain result mismatch");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static void execMMChain(CompressedMatrixBlock cmb, int k) {
		try {
			final int cols = cmb.getNumColumns();
			MatrixBlock v = TestUtils.round(TestUtils.generateTestMatrixBlock(cols, 1, -3, 3, 1.0, 42));
			MatrixBlock uncompressed = CompressedMatrixBlock.getUncompressed(cmb);

			MatrixBlock cRet = cmb.chainMatrixMultOperations(v, null, new MatrixBlock(), ChainType.XtXv, k);
			MatrixBlock uRet = uncompressed.chainMatrixMultOperations(v, null, new MatrixBlock(), ChainType.XtXv, k);

			assertEquals(cols, cRet.getNumRows());
			assertEquals(1, cRet.getNumColumns());
			TestUtils.compareMatrices(uRet, cRet, 1e-6, "mm-chain result mismatch");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
