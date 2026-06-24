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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public final class CLALibSort {

	private CLALibSort() {
		// private constructor for utility class.
	}

	/**
	 * Sort (order) a compressed matrix in place of the {@code order} built-in, while keeping the result compressed.
	 *
	 * The compressed fast-path only supports the case the user can benefit from: a single column held in a single column
	 * group, sorted ascending and returning the sorted values (not the index permutation). For everything else (multiple
	 * columns, multiple column groups, descending order, index return, or a column-group encoding without a sort
	 * implementation) this returns {@code null} so the caller can fall back to a decompressed reorg.
	 *
	 * @param mb the compressed matrix to sort
	 * @param fn the sort specification carried by the reorg operator
	 * @return the sorted compressed matrix, or {@code null} if the compressed fast-path does not apply
	 */
	public static MatrixBlock sort(CompressedMatrixBlock mb, SortIndex fn) {
		final boolean singleColumn = mb.getNumColumns() == 1 && mb.getColGroups().size() == 1;
		if(!singleColumn || fn.getDecreasing() || fn.getIndexReturn())
			return null;

		final AColGroup sorted = sortSingleColumn(mb);
		if(sorted == null)
			return null;

		final List<AColGroup> rg = new ArrayList<>(1);
		rg.add(sorted);
		return new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns(), mb.getNonZeros(), false, rg);
	}

	/**
	 * Compute the sorted value/weight table used by the quantile/median/IQM operations (the {@code sort} / qsort lop),
	 * exploiting compression to sort the few distinct values instead of all rows.
	 *
	 * The compressed fast-path applies to an unweighted sort of a single column held in a single column group. The
	 * produced table is bit-for-bit identical to {@link MatrixBlock#sortOperations(MatrixValue, MatrixBlock, int)}: a
	 * {@code (1 + nnz) x 2} matrix holding one row per non-zero value (weight 1) plus a single collapsed row for the
	 * zeros (weight = number of zeros), sorted ascending by value. For every other case (weights present, multiple
	 * columns or groups, or an encoding without a sort implementation) it falls back to a decompressed sort.
	 *
	 * @param mb      the compressed matrix to sort
	 * @param weights optional per-row weights, or {@code null}
	 * @param result  the result matrix (reused by the fallback)
	 * @param k       the parallelization degree
	 * @return the sorted value/weight table
	 */
	public static MatrixBlock sort(CompressedMatrixBlock mb, MatrixValue weights, MatrixBlock result, int k) {
		final MatrixBlock w = CompressedMatrixBlock.getUncompressed(weights);
		if(w == null && mb.getNumColumns() == 1 && mb.getColGroups().size() == 1) {
			final MatrixBlock fast = sortTableSingleColumn(mb, result, k);
			if(fast != null)
				return fast;
		}

		// fallback to uncompressed sort.
		return CompressedMatrixBlock.getUncompressed(mb, "sortOperations", k).sortOperations(w, result, k);
	}

	private static AColGroup sortSingleColumn(CompressedMatrixBlock mb) {
		try {
			return mb.getColGroups().get(0).sort();
		}
		catch(NotImplementedException e) {
			// the column-group encoding does not implement sort -> let the caller decompress.
			return null;
		}
	}

	private static MatrixBlock sortTableSingleColumn(CompressedMatrixBlock mb, MatrixBlock result, int k) {
		final long lnnz = mb.getNonZeros();
		if(lnnz < 0) // unknown number of non-zeros, cannot size the table.
			return null;

		final AColGroup sorted = sortSingleColumn(mb);
		if(sorted == null)
			return null;

		final int nRows = mb.getNumRows();
		final int nnz = (int) lnnz;
		final int zeroCount = nRows - nnz;

		// decompress the already-sorted single column once (ascending, zeros contiguous).
		final List<AColGroup> rg = new ArrayList<>(1);
		rg.add(sorted);
		final MatrixBlock sortedCol = new CompressedMatrixBlock(nRows, 1, lnnz, false, rg).decompress(k);

		// build the value/weight table: one row per non-zero value, plus a single collapsed zero row.
		final MatrixBlock tdw = new MatrixBlock(1 + nnz, 2, false);
		tdw.allocateDenseBlock();
		int w = 0;
		boolean zeroWritten = false;
		for(int i = 0; i < nRows; i++) {
			final double v = sortedCol.get(i, 0);
			if(v < 0) {
				tdw.set(w, 0, v);
				tdw.set(w, 1, 1);
				w++;
			}
			else {
				if(!zeroWritten) {
					tdw.set(w, 0, 0);
					tdw.set(w, 1, zeroCount);
					w++;
					zeroWritten = true;
				}
				if(v != 0) {
					tdw.set(w, 0, v);
					tdw.set(w, 1, 1);
					w++;
				}
			}
		}
		if(!zeroWritten) { // all values negative: the zero row sorts to the end.
			tdw.set(w, 0, 0);
			tdw.set(w, 1, zeroCount);
		}

		// Emit through the same reorg used by MatrixBlock.sortOperations so the produced table is
		// bit-for-bit identical to the uncompressed path, including its (intentionally unmaintained)
		// non-zero metadata. This keeps downstream quantile/median consumers and result comparisons
		// consistent regardless of whether the input was compressed.
		if(result == null)
			result = new MatrixBlock(1 + nnz, 2, false);
		else
			result.reset(1 + nnz, 2, false);
		final ReorgOperator rop = new ReorgOperator(new SortIndex(1, false, false), k);
		LibMatrixReorg.reorg(tdw, result, rop);
		return result;
	}
}
