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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Random;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Exercises {@code removeEmptyOperations} with a column-selection vector that keeps a strict subset of a
 * <em>multi-column</em> column group. This drives the per-encoding {@code removeEmptyColsSubset} dictionary slicing
 * paths (SDC / SDC-single) and the {@link org.apache.commons.lang3.NotImplementedException} fallback for encodings that
 * do not implement index-only column removal (RLE/OLE).
 */
public class CompressedRemoveEmptyColSubsetTest {

	private static final int ROWS = 500;
	private static final int COLS = 3;

	@Test
	public void colSubsetSDC() {
		// Multiple distinct non-default tuples -> ColGroupSDC.
		runColSubset(buildIdentical(new double[] {3.0, 5.0, 9.0}, 1), CompressionType.SDC);
	}

	@Test
	public void colSubsetSDCSingle() {
		// A single distinct non-default tuple -> ColGroupSDCSingle.
		runColSubset(buildIdentical(new double[] {3.0}, 2), CompressionType.SDC);
	}

	@Test
	public void colSubsetDDC() {
		runColSubset(buildIdentical(new double[] {3.0, 5.0, 9.0}, 3), CompressionType.DDC);
	}

	@Test
	public void colSubsetRLEFallback() {
		runColSubset(buildIdentical(new double[] {3.0, 5.0, 9.0}, 4), CompressionType.RLE);
	}

	@Test
	public void colSubsetOLEFallback() {
		runColSubset(buildIdentical(new double[] {3.0, 5.0, 9.0}, 5), CompressionType.OLE);
	}

	@Test
	public void colSubsetSDCFOR() {
		// SDCFOR cannot be forced at the planner level, so build a multi-column SDC group and sparsify it to the
		// frame-of-reference variant (the production path) before slicing a strict column subset.
		MatrixBlock mb = buildIdentical(new double[] {3.0, 5.0, 9.0}, 8);
		CompressedMatrixBlock sdc = compressForced(mb, CompressionType.SDC);
		AColGroup g = sdc.getColGroups().get(0);
		assumeTrue("Expected a multi-column ColGroupSDC to sparsify", g instanceof ColGroupSDC && g.getNumCols() > 1);
		AColGroup forGroup = ((ColGroupSDC) g).sparsifyFOR();
		assumeTrue("Expected an SDCFOR group after sparsify", forGroup.getCompType() == CompressionType.SDCFOR);

		CompressedMatrixBlock cmb = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), -1, false,
			Collections.singletonList(forGroup));

		MatrixBlock select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);
		MatrixBlock actual = cmb.removeEmptyOperations(null, false, false, select);
		select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);
		MatrixBlock expected = mb.removeEmptyOperations(null, false, false, select);
		TestUtils.compareMatrices(expected, actual, 0.0, "removeEmpty col subset for SDCFOR");
	}

	/** Column selection vector with unknown (-1) non-zero count, forcing the recompute branch. */
	@Test
	public void colSubsetUnknownNnz() {
		MatrixBlock mb = buildIdentical(new double[] {3.0, 5.0, 9.0}, 6);
		CompressedMatrixBlock cmb = compressForced(mb, CompressionType.SDC);
		MatrixBlock select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);
		select.setNonZeros(-1);
		MatrixBlock actual = cmb.removeEmptyOperations(null, false, false, select);
		select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);
		MatrixBlock expected = mb.removeEmptyOperations(null, false, false, select);
		TestUtils.compareMatrices(expected, actual, 0.0, "removeEmpty cols unknown-nnz select");
	}

	/** Row selection vector with unknown (-1) non-zero count, forcing the recompute branch. */
	@Test
	public void rowsUnknownNnz() {
		MatrixBlock mb = buildIdentical(new double[] {3.0, 5.0, 9.0}, 7);
		CompressedMatrixBlock cmb = compressForced(mb, CompressionType.SDC);
		MatrixBlock select = rowSelect();
		select.setNonZeros(-1);
		MatrixBlock actual = cmb.removeEmptyOperations(null, true, false, select);
		MatrixBlock expected = mb.removeEmptyOperations(null, true, false, rowSelect());
		TestUtils.compareMatrices(expected, actual, 0.0, "removeEmpty rows unknown-nnz select");
	}

	private void runColSubset(MatrixBlock mb, CompressionType ct) {
		CompressedMatrixBlock cmb = compressForced(mb, ct);
		assertTrue("Expected a multi-column " + ct + " group to reach the subset path",
			cmb.getColGroups().stream().anyMatch(g -> g.getNumCols() > 1));

		// Keep a strict subset (drop the middle column) so removeEmptyColsSubset is hit instead of copyAndSet.
		MatrixBlock select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);

		MatrixBlock actual = cmb.removeEmptyOperations(null, false, false, select);
		select = new MatrixBlock(1, COLS, false);
		select.set(0, 0, 1);
		select.set(0, 2, 1);
		MatrixBlock expected = mb.removeEmptyOperations(null, false, false, select);
		TestUtils.compareMatrices(expected, actual, 0.0, "removeEmpty col subset for " + ct);
	}

	private static MatrixBlock rowSelect() {
		MatrixBlock select = new MatrixBlock(ROWS, 1, false);
		for(int i = 0; i < ROWS; i += 2)
			select.set(i, 0, 1);
		return select;
	}

	private static CompressedMatrixBlock compressForced(MatrixBlock mb, CompressionType ct) {
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0)
			.setValidCompressions(EnumSet.of(ct));
		MatrixBlock c = CompressedMatrixBlockFactory.compress(mb, 1, csb).getLeft();
		assertTrue("Expected the input to compress into a " + ct + " backed block", c instanceof CompressedMatrixBlock);
		return (CompressedMatrixBlock) c;
	}

	/**
	 * Builds a {@code ROWS x COLS} matrix whose columns are identical so column co-coding merges them into a single
	 * multi-column group, with one dominant value plus a few off-values.
	 */
	private static MatrixBlock buildIdentical(double[] others, int seed) {
		MatrixBlock mb = new MatrixBlock(ROWS, COLS, false);
		mb.allocateDenseBlock();
		Random r = new Random(seed);
		for(int i = 0; i < ROWS; i++) {
			double v = 7.0;
			if(r.nextDouble() < 0.2)
				v = others[r.nextInt(others.length)];
			for(int j = 0; j < COLS; j++)
				mb.set(i, j, v);
		}
		mb.recomputeNonZeros();
		return mb;
	}
}
