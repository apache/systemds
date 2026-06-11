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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.lib.CLALibRightMultBy;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Right matrix multiply on compressed inputs that contain SDC / SDC-zeros column groups.
 *
 * <p>
 * The PR stops forcing a decompressing right multiply for {@link ASDC} / {@link ASDCZero} backed inputs (they have
 * working pre-aggregate paths). These tests build such inputs and verify the compressed right multiply still matches
 * the uncompressed reference for both single-threaded and parallel execution.
 * </p>
 */
public class CLALibRightMultBySDCTest {
	protected static final Log LOG = LogFactory.getLog(CLALibRightMultBySDCTest.class.getName());

	@BeforeClass
	public static void setup() {
		Thread.currentThread().setName("main_test_" + Thread.currentThread().getId());
	}

	/**
	 * Build a compressed matrix dominated by a single value with a handful of exceptions per column, which compresses
	 * into SDC / SDC-zeros column groups.
	 */
	private static CompressedMatrixBlock sdcBlock(int rows, int cols, double sparsity, int seed) {
		MatrixBlock mb = TestUtils.round(TestUtils.generateTestMatrixBlock(rows, cols, 1, 5, sparsity, seed));
		CompressedMatrixBlock cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
		return cmb;
	}

	private static boolean containsSDC(CompressedMatrixBlock cmb) {
		for(AColGroup g : cmb.getColGroups())
			if(g instanceof ASDC || g instanceof ASDCZero)
				return true;
		return false;
	}

	@Test
	public void rightMultVectorSparseSingleThread() {
		execRightMult(sdcBlock(500, 6, 0.2, 21), 1, 1);
	}

	@Test
	public void rightMultVectorSparseParallel() {
		execRightMult(sdcBlock(500, 6, 0.2, 22), 1, 4);
	}

	@Test
	public void rightMultMatrixSparseSingleThread() {
		execRightMult(sdcBlock(500, 6, 0.2, 23), 4, 1);
	}

	@Test
	public void rightMultMatrixSparseParallel() {
		execRightMult(sdcBlock(500, 6, 0.2, 24), 4, 4);
	}

	@Test
	public void rightMultWideSparseParallel() {
		execRightMult(sdcBlock(500, 6, 0.2, 27), 12, 4);
	}

	private static void execRightMult(CompressedMatrixBlock cmb, int rhsCols, int k) {
		try {
			assertTrue("test input should contain an SDC/SDCZeros column group", containsSDC(cmb));

			final int cols = cmb.getNumColumns();
			MatrixBlock right = TestUtils.round(TestUtils.generateTestMatrixBlock(cols, rhsCols, -3, 3, 1.0, 99));
			MatrixBlock uncompressed = CompressedMatrixBlock.getUncompressed(cmb);

			MatrixBlock cRet = CLALibRightMultBy.rightMultByMatrix(cmb, right, null, k);
			MatrixBlock uRet = LibMatrixMult.matrixMult(uncompressed, right, k);

			TestUtils.compareMatricesBitAvgDistance(uRet, CompressedMatrixBlock.getUncompressed(cRet), 1024, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
