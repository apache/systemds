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

package org.apache.sysds.runtime.compress.estim.encoding;

import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * This interface covers an intermediate encoding for the samples to improve the efficiency of the joining of sample
 * column groups.
 */
public interface IEncode {
	static final Log LOG = LogFactory.getLog(IEncode.class.getName());

	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, int[] rowCols) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(rowCols.length == 1)
			return createFromMatrixBlock(m, transposed, rowCols[0]);
		else
			return createWithReader(m, rowCols, transposed);
	}

	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, int[] rowCols){
		return createFromMatrixBlockDelta(m, transposed, rowCols, transposed ? m.getNumColumns() : m.getNumRows());
	}

	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, int[] rowCols, int nVals){
		throw new NotImplementedException();
	}

	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, int rowCol) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(transposed) {
			if(m.isInSparseFormat())
				return createFromSparseTransposed(m, rowCol);
			else
				return createFromDenseTransposed(m, rowCol);
		}
		else if(m.isInSparseFormat())
			return createFromSparse(m, rowCol);
		else
			return createFromDense(m, rowCol);
	}

	public static IEncode createFromDenseTransposed(MatrixBlock m, int row) {
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final DenseBlock db = m.getDenseBlock();
		final int off = db.pos(row);
		final int nCol = m.getNumColumns();
		final int end = off + nCol;
		final double[] vals = db.values(row);

		// Iteration 1, make Count HashMap.
		for(int i = off; i < end; i++) // sequential access
			map.increment(vals[i]);

		final int nUnique = map.size();

		if(nUnique == 1)
			return new ConstEncoding(m.getNumColumns());

		if(map.getOrDefault(0, -1) * 10 > nCol * 4) { // 40 %
			final int[] counts = map.getUnorderedCountsAndReplaceWithUIDsWithout0(); // map.getUnorderedCountsAndReplaceWithUIDs();
			final int zeroCount = map.get(0);
			final int nV = nCol - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique - 1);

			// for(int i = off, r = 0, di = 0; i < end; i += nCol, r++){
			for(int i = off, r = 0, di = 0; i < end; i++, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.get(vals[i]) );
				}
			}

			final AOffset o = OffsetFactory.createOffset(offsets);
			return new SparseEncoding(d, o, zeroCount, counts, nCol);
		}
		else {
			// Allocate counts, and iterate once to replace counts with u ids
			final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();

			// Create output map
			final AMapToData d = MapToFactory.create(nCol, nUnique);

			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i++, r++)
				d.set(r, map.get(vals[i]));

			return new DenseEncoding(d, counts);
		}
	}

	public static IEncode createFromSparseTransposed(MatrixBlock m, int row) {
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final SparseBlock sb = m.getSparseBlock();
		if(sb.isEmpty(row))
			return new EmptyEncoding();
		final int apos = sb.pos(row);
		final int alen = sb.size(row) + apos;
		final double[] avals = sb.values(row);

		// Iteration 1 of non zero values, make Count HashMap.
		for(int i = apos; i < alen; i++) // sequential of non zero cells.
			map.increment(avals[i]);

		final int nUnique = map.size();

		// Allocate counts
		final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();

		// Create output map
		final AMapToData d = MapToFactory.create(alen - apos, nUnique);

		// Iteration 2 of non zero values, make either a IEncode Dense or sparse map.
		for(int i = apos, j = 0; i < alen; i++, j++)
			d.set(j, map.get(avals[i]));

		// Iteration 3 of non zero indexes, make a Offset Encoding to know what cells are zero and not.
		// not done yet
		AOffset o = OffsetFactory.createOffset(sb.indexes(row), apos, alen);

		final int zero = m.getNumColumns() - o.getSize();
		return new SparseEncoding(d, o, zero, counts, m.getNumColumns());
	}

	public static IEncode createFromDense(MatrixBlock m, int col) {
		final DenseBlock db = m.getDenseBlock();
		if(!db.isContiguous())
			throw new NotImplementedException("Not Implemented non contiguous dense matrix encoding for sample");
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final int off = col;
		final int nCol = m.getNumColumns();
		final int nRow = m.getNumRows();
		final int end = off + nRow * nCol;
		final double[] vals = m.getDenseBlockValues();

		// Iteration 1, make Count HashMap.
		for(int i = off; i < end; i += nCol) // jump down through rows.
			map.increment(vals[i]);

		final int nUnique = map.size();
		if(nUnique == 1)
			return new ConstEncoding(m.getNumColumns());

		if(map.getOrDefault(0, -1) * 10 > nRow * 4) { // 40 %
			final int[] counts = map.getUnorderedCountsAndReplaceWithUIDsWithout0();
			final int zeroCount = map.get(0);
			final int nV = m.getNumRows() - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique);

			for(int i = off, r = 0, di = 0; i < end; i += nCol, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.get(vals[i]));
				}
			}

			final AOffset o = OffsetFactory.createOffset(offsets);

			return new SparseEncoding(d, o, zeroCount, counts, nRow);
		}
		else {
			// Allocate counts, and iterate once to replace counts with u ids
			final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();

			final AMapToData d = MapToFactory.create(nRow, nUnique);
			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i += nCol, r++)
				d.set(r, map.get(vals[i]));
			return new DenseEncoding(d, counts);
		}
	}

	public static IEncode createFromSparse(MatrixBlock m, int col) {

		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final SparseBlock sb = m.getSparseBlock();

		final double guessedNumberOfNonZero = Math.min(4, Math.ceil((double)m.getNumRows() * m.getSparsity()));
		final IntArrayList offsets = new IntArrayList((int)guessedNumberOfNonZero);

		// Iteration 1 of non zero values, make Count HashMap.
		for(int r = 0; r < m.getNumRows(); r++) { // Horrible performance but ... it works.
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			final int index = Arrays.binarySearch(aix, apos, alen, col);
			if(index >= 0) {
				offsets.appendValue(r);
				map.increment(sb.values(r)[index]);
			}
		}
		if(offsets.size() == 0)
			return new EmptyEncoding();

		final int nUnique = map.size();
		final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();

		int sumCounts = 0;
		for(int c : counts)
			sumCounts += c;

		final AMapToData d = MapToFactory.create(sumCounts, nUnique);

		// Iteration 2 of non zero values, make either a IEncode Dense or sparse map.
		for(int off = 0, r = 0; off < sumCounts; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			// Performance hit because of binary search for each row.
			final int index = Arrays.binarySearch(aix, apos, alen, col);
			if(index >= 0)
				d.set(off++, map.get(sb.values(r)[index]));
		}

		// Iteration 3 of non zero indexes, make a Offset Encoding to know what cells are zero and not.
		AOffset o = OffsetFactory.createOffset(offsets);

		final int zero = m.getNumRows() - sumCounts;
		return new SparseEncoding(d, o, zero, counts, m.getNumRows());
	}

	public static IEncode createWithReader(MatrixBlock m, int[] rowCols, boolean transposed) {
		final ReaderColumnSelection reader1 = ReaderColumnSelection.createReader(m, rowCols, transposed);
		final int nRows = transposed ? m.getNumColumns() : m.getNumRows();
		final DblArrayCountHashMap map = new DblArrayCountHashMap(16, rowCols.length);
		final IntArrayList offsets = new IntArrayList();
		DblArray cellVals = reader1.nextRow();

		// Iteration 1, make Count HashMap, and offsets.
		while(cellVals != null) {
			map.increment(cellVals);
			offsets.appendValue(reader1.getCurrentRowIndex());
			cellVals = reader1.nextRow();
		}

		if(offsets.size() == 0)
			return new EmptyEncoding();

		if(map.size() == 1 && offsets.size() == nRows)
			return new ConstEncoding(nRows);

		if(offsets.size() < nRows) {
			// there was fewer offsets than rows.
			if(offsets.size() < nRows / 2) {
				// Output encoded Sparse since there is more than half empty.
				final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();
				final int zeros = nRows - offsets.size();
				return createWithReaderSparse(m, map, zeros, counts, rowCols, offsets, nRows, transposed);
			}
			else {
				// Output Encoded dense since there is not enough common values.
				// TODO add Common group, that allows to now allocate this extra cell
				final int[] counts = map.getUnorderedCountsAndReplaceWithUIDsWithExtraCell();
				counts[counts.length - 1] = nRows - offsets.size();
				return createWithReaderDense(m, map, counts, rowCols, nRows, transposed);
			}
		}
		else {
			// TODO add Common group, that allows to allocate with one of the map entries as the common value.
			// the input was fully dense.

			final int[] counts = map.getUnorderedCountsAndReplaceWithUIDs();
			return createWithReaderDense(m, map, counts, rowCols, nRows, transposed);
		}
	}

	public static IEncode createWithReaderDense(MatrixBlock m, DblArrayCountHashMap map, int[] counts, int[] rowCols,
		int nRows, boolean transposed) {
		// Iteration 2,
		final ReaderColumnSelection reader2 = ReaderColumnSelection.createReader(m, rowCols, transposed);
		final AMapToData d = MapToFactory.create(nRows, counts.length);
		final int def = counts.length - 1;

		DblArray cellVals = reader2.nextRow();
		int r = 0;
		while(r < nRows && cellVals != null) {
			final int row = reader2.getCurrentRowIndex();
			if(row == r) {
				d.set(row, map.get(cellVals));
				cellVals = reader2.nextRow();
			}
			else
				d.set(r, def);
			r++;
		}

		while(r < nRows)
			d.set(r++, def);
		return new DenseEncoding(d, counts);
	}

	public static IEncode createWithReaderSparse(MatrixBlock m, DblArrayCountHashMap map, int zeros, int[] counts,
		int[] rowCols, IntArrayList offsets, int nRows, boolean transposed) {
		final ReaderColumnSelection reader2 = ReaderColumnSelection.createReader(m, rowCols, transposed);
		DblArray cellVals = reader2.nextRow();

		final AMapToData d = MapToFactory.create(offsets.size(), map.size());

		int i = 0;
		// Iterator 2 of non zero tuples.
		while(cellVals != null) {
			d.set(i++, map.get(cellVals));
			cellVals = reader2.nextRow();
		}

		// iteration 3 of non zero indexes,
		final AOffset o = OffsetFactory.createOffset(offsets);

		return new SparseEncoding(d, o, zeros, counts, nRows);
	}

	public IEncode join(IEncode e);

	public int getUnique();

	public int size();

	public int[] getCounts();

	public EstimationFactors computeSizeEstimation(int[] cols, int nRows, double tupleSparsity, double matrixSparsity);

}
