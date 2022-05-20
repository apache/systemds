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

package org.apache.sysds.test.component.compress.readers;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseMultiBlock;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseMultiBlockTransposed;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseSingleBlock;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseSingleBlockTransposed;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionSparse;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionSparseTransposed;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class ReadersTestCompareReaders {

	protected static final Log LOG = LogFactory.getLog(ReadersTestCompareReaders.class.getName());

	public final MatrixBlock m;
	public final MatrixBlock sm;
	public final MatrixBlock tm;
	public final MatrixBlock tsm;
	public final MatrixBlock mMockLarge;
	public final MatrixBlock mMockLargeTransposed;

	public final int[] cols;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		MatrixBlock mb;

		final double[] spar = new double[] {1.0, 0.9, 0.3, 0.1, 0.01};
		final int[] cols = new int[] {16, 10, 6, 4, 3, 2};
		final int[] rows = new int[] {1423, 1000, 500};
		for(int i = 0; i < 3; i++) { // seeds
			for(int s = 0; s < spar.length; s++) {
				for(int c = 0; c < cols.length; c++) {
					for(int r = 0; r < rows.length; r++) {
						mb = TestUtils.generateTestMatrixBlock(rows[r], cols[c], 1, 10, spar[s], i);
						tests.add(new Object[] {mb});
					}
				}
			}
		}

		// only one row is set
		mb = new MatrixBlock(100, 10, false);
		for(int i = 0; i < 10; i++)
			mb.quickSetValue(3, i, 231);
		tests.add(new Object[] {mb});

		// only one col is set
		mb = new MatrixBlock(100, 10, false);
		for(int i = 0; i < 100; i++)
			mb.quickSetValue(i, 4, 231);
		tests.add(new Object[] {mb});

		return tests;
	}

	public ReadersTestCompareReaders(MatrixBlock in) {

		if(in.getNumColumns() > 9)
			cols = createColIdx(4, 8, 2);
		else if(in.getNumColumns() > 3)
			cols = createColIdx(1, 3);
		else // take first two columns.
			cols = createColIdx(0, 2);

		final boolean isSparseIn = in.isInSparseFormat();
		MatrixBlock m2 = new MatrixBlock();
		m2.copy(in);
		if(isSparseIn) {
			sm = in;
			m2.sparseToDense();
			m = m2;
		}
		else {
			m = in;
			m2.denseToSparse(true);
			sm = m2;
		}

		MatrixBlock tin = LibMatrixReorg.transpose(in);
		MatrixBlock m2t = new MatrixBlock();
		m2t.copy(tin);
		final boolean isSparseTIn = in.isInSparseFormat();
		if(isSparseTIn) {
			tsm = tin;
			m2t.sparseToDense();
			tm = m2t;
		}
		else {
			tm = tin;
			m2t.denseToSparse(true);
			tsm = m2t;
		}

		mMockLarge = new MatrixBlock(m.getNumRows(), m.getNumColumns(),
			new DenseBlockFP64Mock(new int[] {m.getNumRows(), m.getNumColumns()}, m.getDenseBlockValues()));

		mMockLargeTransposed = new MatrixBlock(tm.getNumRows(), tm.getNumColumns(),
			new DenseBlockFP64Mock(new int[] {tm.getNumRows(), tm.getNumColumns()}, tm.getDenseBlockValues()));

	}

	@Test
	public void testCompareSparseDense() {
		ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false);
		ReaderColumnSelection b = ReaderColumnSelection.createReader(sm, cols, false);
		if(b instanceof ReaderColumnSelectionSparse && a instanceof ReaderColumnSelectionDenseSingleBlock)
			compareReaders(a, b);
		else
			fail("Incorrect type of reader");
	}

	@Test
	public void testCompareSparseDenseFewRowsIn() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = m.getNumRows() - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(sm, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareSparseDenseFewRowsFromEnd() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int end = m.getNumRows() - 10;
			final int start = end - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(sm, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareSparseDenseFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = start + 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(sm, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareSparseDenseLastFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = m.getNumRows() - 5;
			final int end = m.getNumRows() - 1;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(sm, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedDense() {
		ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false);
		ReaderColumnSelection b = ReaderColumnSelection.createReader(tm, cols, true);
		if(b instanceof ReaderColumnSelectionDenseSingleBlockTransposed)
			compareReaders(a, b);
		else
			fail("Incorrect type of reader");
	}

	@Test
	public void testCompareDenseTransposedDenseFewRowsIn() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = m.getNumRows() - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedDenseFewRowsFromEnd() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int end = m.getNumRows() - 10;
			final int start = end - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedDenseFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = start + 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparse() {
		ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false);
		ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true);
		if(b instanceof ReaderColumnSelectionSparseTransposed)
			compareReaders(a, b);
		else
			fail("Incorrect type of reader");
	}

	@Test
	public void testCompareDenseTransposedSparseFewRowsIn() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = m.getNumRows() - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseFewRowsFromEnd() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int end = m.getNumRows() - 10;
			final int start = end - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = start + 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseToManyRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = nRow + 10; // to large end ... but it should correct itself.
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseSingleRow() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 26;
			final int end = 27;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseLastRow() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = nRow - 1;
			final int end = nRow;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseBasedOnValueOffsets() {
		final SparseBlock sb = tsm.getSparseBlock();
		final int[] idx = sb.indexes(cols[0]);
		final int apos = sb.pos(cols[0]);
		final int alen = sb.size(cols[0]) + apos;
		if(alen - apos > 2) {
			final int end = idx[idx.length - 1];
			final int start = Math.max(0, end - 2);
			if(end > start) {
				ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
				ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
				compareReaders(a, b, start, end);
			}
		}
	}

	@Test
	public void testCompareDenseTransposedSparseBasedOnValueOffsetsTwoLast() {
		SparseBlock sb = tsm.getSparseBlock();
		final int[] idx = sb.indexes(cols[0]);
		final int apos = sb.pos(cols[0]);
		final int alen = sb.size(cols[0]) + apos;
		if(alen - apos > 2) {

			final int end = idx[alen - 1];
			final int start = idx[alen - 2];
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseBasedOnValueOffsetsOnLast() {
		SparseBlock sb = tsm.getSparseBlock();
		final int[] idx = sb.indexes(cols[0]);
		final int apos = sb.pos(cols[0]);
		final int alen = sb.size(cols[0]) + apos;
		if(alen - apos > 2) {

			final int end = tsm.getNumColumns();
			final int start = idx[alen - 1];
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseBasedOnValueOffsetsTwoFirst() {
		SparseBlock sb = tsm.getSparseBlock();
		final int[] idx = sb.indexes(cols[0]);
		final int apos = sb.pos(cols[0]);
		final int alen = sb.size(cols[0]) + apos;
		if(alen - apos > 2) {

			final int start = idx[apos];
			final int end = idx[apos + 1];
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedSparseSecondLastRow() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = nRow - 2;
			final int end = nRow - 1;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(tsm, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseLarge() {
		ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false);
		ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLarge, cols, false);
		if(b instanceof ReaderColumnSelectionDenseMultiBlock)
			compareReaders(a, b);
		else
			fail("Incorrect reader type");
	}

	@Test
	public void testCompareDenseLargeFewRowsIn() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = m.getNumRows() - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLarge, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseLargeFewRowsFromEnd() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int end = m.getNumRows() - 10;
			final int start = end - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLarge, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseLargeFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = start + 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLarge, cols, false, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedLarge() {
		ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false);
		ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLargeTransposed, cols, true);
		if(b instanceof ReaderColumnSelectionDenseMultiBlockTransposed)
			compareReaders(a, b);
		else
			fail("Incorrect reader type");
	}

	@Test
	public void testCompareDenseTransposedLargeFewRowsIn() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = m.getNumRows() - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLargeTransposed, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedLargeFewRowsFromEnd() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int end = m.getNumRows() - 10;
			final int start = end - 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLargeTransposed, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	@Test
	public void testCompareDenseTransposedLargeFewRows() {
		final int nRow = m.getNumRows();
		if(nRow > 30) {
			final int start = 10;
			final int end = start + 10;
			ReaderColumnSelection a = ReaderColumnSelection.createReader(m, cols, false, start, end);
			ReaderColumnSelection b = ReaderColumnSelection.createReader(mMockLargeTransposed, cols, true, start, end);
			compareReaders(a, b, start, end);
		}
	}

	private void compareReaders(ReaderColumnSelection a, ReaderColumnSelection b) {
		try {

			DblArray ar = null;
			DblArray br = null;
			while((ar = a.nextRow()) != null) {
				br = b.nextRow();
				final int aIdx = a.getCurrentRowIndex();
				final int bIdx = b.getCurrentRowIndex();
				if(aIdx != bIdx)
					fail("Not equal row indexes" + aIdx + "  " + bIdx);
				if(!ar.equals(br))
					fail("Not equal row values" + ar + " " + br + " at row: " + aIdx);
			}
			br = b.nextRow();

			if(ar != null || br != null)
				fail("both iterators were not done " + a.getCurrentRowIndex() + " " + b.getCurrentRowIndex() + " " + ar
					+ " " + br + "  " + a.getClass().getSimpleName() + "  " + b.getClass().getSimpleName());

			ar = a.nextRow();
			br = b.nextRow();
			if(ar != null || br != null)
				fail("both iterators still return null " + a.getCurrentRowIndex() + " " + b.getCurrentRowIndex());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Exception thrown while iterating");
		}

	}

	private void compareReaders(final ReaderColumnSelection a, final ReaderColumnSelection b, final int start,
		final int end) {
		try {
			DblArray ar = null;
			DblArray br = null;
			while((ar = a.nextRow()) != null) {
				br = b.nextRow();
				final int aIdx = a.getCurrentRowIndex();
				final int bIdx = b.getCurrentRowIndex();

				if(aIdx != bIdx)
					fail("Not equal row indexes" + aIdx + "  " + bIdx);
				if(aIdx < start)
					fail("reader violated the row lower");
				if(aIdx >= end)
					fail("reader violated the row upper " + aIdx + " is larger or equal to " + end + "  "
						+ b.getClass().getSimpleName());
				if(!ar.equals(br))
					fail("Not equal row values" + ar + " " + br + " at row: " + aIdx);
			}
			br = b.nextRow();

			if(ar != null || br != null)
				fail("both iterators were not done " + a.getCurrentRowIndex() + " " + b.getCurrentRowIndex());

			ar = a.nextRow();
			br = b.nextRow();
			if(ar != null || br != null)
				fail("both iterators still return null " + a.getCurrentRowIndex() + " " + b.getCurrentRowIndex());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Exception thrown while iterating");
		}
	}

	private int[] createColIdx(int start, int end) {
		int[] subCols = new int[end - start];
		for(int i = start; i < end; i++)
			subCols[i - start] = i;
		return subCols;
	}

	private int[] createColIdx(int start, int end, int skip) {
		int[] subCols = new int[(end - start) / skip];
		for(int i = start; i < end; i += skip)
			subCols[(i - start) / skip] = i;
		return subCols;
	}

	private class DenseBlockFP64Mock extends DenseBlockFP64 {
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
