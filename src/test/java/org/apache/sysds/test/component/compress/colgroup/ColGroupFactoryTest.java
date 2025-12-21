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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.cost.DistinctCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class ColGroupFactoryTest {

	private final MatrixBlock mb;
	private final MatrixBlock mbt;
	private final ACostEstimate ce;
	private final CompressionSettingsBuilder csb;
	private final int nRow;
	private final int nCol;
	private final CompressionType ct;
	private final List<AColGroup> g;

	private final IColIndex cols;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		try {

			// add(tests, 40, 5, 2, 5, 0.7, 234);
			// add(tests, 40, 5, 1, 1, 0.7, 234);
			add(tests, 40, 1, 2, 5, 0.7, 234);
			add(tests, 40, 1, 1, 1, 0.7, 234);
			add(tests, 40, 5, 2, 5, 0.2, 234);
			add(tests, 40, 5, 2, 5, 0.1, 234);
			add(tests, 40, 1, 2, 5, 0.1, 234);
			add(tests, 40, 1, 1, 1, 0.8, 234);
			add(tests, 40, 1, 1, 1, 0.1, 234);
			add(tests, 40, 5, 2, 5, 0.0, 234);
			add(tests, 40, 1, 1, 3, 1.0, 234);
			add(tests, 40, 1, 1, 3, 1.0, 234);

			addWithEmpty(tests, 40, 1, 2, 5, 0.1, 234);
			addWithEmpty(tests, 40, 1, 1, 1, 0.1, 234);
			addWithEmpty(tests, 40, 1, 1, 1, 0.8, 234);
			addWithEmpty(tests, 40, 3, 2, 5, 0.1, 234);

			addWithEmptyReverse(tests, 40, 1, 2, 5, 0.1, 234);
			addWithEmptyReverse(tests, 40, 1, 2, 5, 0.7, 234);
			addWithEmptyReverse(tests, 40, 1, 1, 1, 0.7, 234);
			addWithEmptyReverse(tests, 40, 3, 2, 5, 0.1, 234);

			addDenseMultiBlock(tests, 40, 3, 2, 5, 0.7, 234);
			addDenseMultiBlock(tests, 40, 1, 2, 5, 0.7, 234);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		return tests;
	}

	private static void addDenseMultiBlock(ArrayList<Object[]> tests, int nRows, int nCols, int min, int max,
		double sparsity, int seed) {
		if(nCols <= 1)
			nCols += 1;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRows, nCols, min, max, sparsity, seed);
		mb = TestUtils.ceil(mb);

		MatrixBlock mbt = LibMatrixReorg.transpose(mb);

		mb = new MatrixBlock(mb.getNumRows(), mb.getNumColumns(),
			new DenseBlockFP64Mock(new int[] {mb.getNumRows(), mb.getNumColumns()}, mb.getDenseBlockValues()));
		mbt = new MatrixBlock(mbt.getNumRows(), mbt.getNumColumns(),
			new DenseBlockFP64Mock(new int[] {mbt.getNumRows(), mbt.getNumColumns()}, mbt.getDenseBlockValues()));

		add(tests, nCols , mb, mbt);
	}

	private static void addWithEmpty(ArrayList<Object[]> tests, int nRows, int nCols, int min, int max, double sparsity,
		int seed) {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRows, nCols, min, max, sparsity, seed);
		mb = TestUtils.ceil(mb);

		mb = mb.append(new MatrixBlock(nRows, 3, false), null);

		MatrixBlock mbt = LibMatrixReorg.transpose(mb);

		mb.denseToSparse(true);
		mbt.denseToSparse(true);

		add(tests, nCols + 3, mb, mbt);
	}

	private static void addWithEmptyReverse(ArrayList<Object[]> tests, int nRows, int nCols, int min, int max,
		double sparsity, int seed) {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRows, nCols, min, max, sparsity, seed);
		mb = TestUtils.ceil(mb);

		mb = new MatrixBlock(nRows, 3, false).append(mb, null);

		MatrixBlock mbt = LibMatrixReorg.transpose(mb);

		mb.denseToSparse(true);
		mbt.denseToSparse(true);

		add(tests, nCols + 3, mb, mbt);
	}

	private static void add(ArrayList<Object[]> tests, int nRows, int nCols, int min, int max, double sparsity,
		int seed) {

		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRows, nCols, min, max, sparsity, seed);
		mb = TestUtils.ceil(mb);
		MatrixBlock mbt = LibMatrixReorg.transpose(mb);

		if(sparsity < 0.4) {
			mb.denseToSparse(true);
			mbt.denseToSparse(true);
		}
		add(tests, nCols, mb, mbt);
	}

	private static void add(ArrayList<Object[]> tests, int nCols, MatrixBlock mb, MatrixBlock mbt) {
		final CompressionSettingsBuilder csb = new CompressionSettingsBuilder();

		ACostEstimate cce = new ComputationCostEstimator(2, 2, 2, 2, 2, 2, 2, 2, true);
		ACostEstimate dce = new DistinctCostEstimator(mb.getNumRows(), csb.create(), mb.getSparsity());

		IColIndex cols = ColIndexFactory.create(nCols);
		IColIndex cols2 = nCols > 1 ? ColIndexFactory.create(1, nCols) : null;
		IColIndex cols3 = ColIndexFactory.create(new int[] {nCols - 1});
		try {
			for(CompressionType ct : CompressionType.values()) {
				if(ct == CompressionType.DeltaDDC)
					continue;
				for(ACostEstimate ce : new ACostEstimate[] {null, cce, dce}) {

					tests.add(new Object[] {mb, mbt, ce, csb, ct, cols});
					if(nCols > 1) {
						tests.add(new Object[] {mb, mbt, ce, csb, ct, cols2});
						tests.add(new Object[] {mb, mbt, ce, csb, ct, cols3});
					}
				}

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}
	}

	public ColGroupFactoryTest(MatrixBlock mb, MatrixBlock mbt, ACostEstimate ce, CompressionSettingsBuilder csb,
		CompressionType ct, IColIndex cols) {
		this.mb = mb;
		this.nRow = mb.getNumRows();
		this.nCol = mb.getNumColumns();
		this.ce = ce;
		this.csb = csb;
		this.mbt = mbt;
		this.ct = ct;
		this.cols = cols;
		g = compST();
	}

	@Test
	public void testCompressTransposedSingleThread() {
		compare(compTransposedST());
	}

	@Test
	public void testCompressTransposedMultiThread() {
		compare(compTransposedMT());
	}

	@Test
	public void testCompressMultiThread() {
		compare(compMT());
	}

	@Test
	public void testCompressMultiThreadDDC() {
		if(ct == CompressionType.DDC) {
			CompressionSettings.PAR_DDC_THRESHOLD = 1;
			compare(compMT());
			compare(compST());
			CompressionSettings.PAR_DDC_THRESHOLD = 13425;
		}
	}

	@Test
	public void testCompressMultipleTimes() {
		try {

			final int offs = Math.min((int) (mbt.getSparsity() * nRow * nCol), nRow);
			final EstimationFactors f = new EstimationFactors(Math.min(nRow, offs), nRow, offs, mbt.getSparsity());
			final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
			es.add(new CompressedSizeInfoColGroup(cols, f, 312152, ct));
			es.add(new CompressedSizeInfoColGroup(cols, f, 312152, ct));// second time.
			final CompressedSizeInfo csi = new CompressedSizeInfo(es);
			CompressionSettings cs = csb.create();

			cs.transposed = true;
			if(ce != null)
				ColGroupFactory.compressColGroups(mbt, csi, cs, ce, 4);
			else
				ColGroupFactory.compressColGroups(mbt, csi, cs, 4);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void compare(List<AColGroup> gt) {
		for(int i = 0; i < g.size(); i++)
			compare(gt.get(i), g.get(i));
	}

	private void compare(AColGroup gtt, AColGroup gg) {
		assertEquals(gtt.getColIndices(), gg.getColIndices());
	}

	private List<AColGroup> compST() {
		return comp(1, false);
	}

	private List<AColGroup> compMT() {
		return comp(4, false);
	}

	private List<AColGroup> compTransposedST() {
		return comp(1, true);
	}

	private List<AColGroup> compTransposedMT() {
		return comp(4, true);
	}

	private List<AColGroup> comp(int k, boolean transposed) {
		try {
			final int offs = Math.min((int) (mbt.getSparsity() * nRow * nCol), nRow);
			final EstimationFactors f = new EstimationFactors(Math.min(nRow, offs), nRow, offs, mbt.getSparsity());
			final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
			es.add(new CompressedSizeInfoColGroup(cols, f, 314152, ct));
			final CompressedSizeInfo csi = new CompressedSizeInfo(es);
			CompressionSettings cs = csb.create();

			if(transposed) {
				cs.transposed = true;
				if(ce != null)
					return ColGroupFactory.compressColGroups(mbt, csi, cs, ce, 1);
				else
					return ColGroupFactory.compressColGroups(mbt, csi, cs, 1);
			}
			else {
				if(ce != null)
					return ColGroupFactory.compressColGroups(mb, csi, cs, ce, k);
				else if(k == 1)
					return ColGroupFactory.compressColGroups(mb, csi, cs);
				else
					return ColGroupFactory.compressColGroups(mb, csi, cs, k);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to compress");
			return null;
		}
	}

	private static class DenseBlockFP64Mock extends DenseBlockFP64 {
		private static final long serialVersionUID = -3601232958390554672L;

		public DenseBlockFP64Mock(int[] dims, double[] data) {
			super(dims, data);
		}

		@Override
		public boolean isContiguous() {
			return false;
		}

	@Override
	public int numBlocks() {
		return 2;
	}
}
}
