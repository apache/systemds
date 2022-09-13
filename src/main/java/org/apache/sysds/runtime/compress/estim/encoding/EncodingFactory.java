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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface EncodingFactory {

	/**
	 * Encode a list of columns together from the input matrix, as if it is cocoded.
	 * 
	 * @param m          The matrix input to encode
	 * @param transposed If the matrix is transposed in memory
	 * @param rowCols    The list of columns to encode.
	 * @return An encoded format of the information of the columns.
	 */
	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, int[] rowCols) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(rowCols.length == 1)
			return createFromMatrixBlock(m, transposed, rowCols[0]);
		else
			return createWithReader(m, rowCols, transposed);
	}

	/**
	 * Encode a full delta representation of the matrix input taking all rows into account.
	 * 
	 * Note the input matrix should not be delta encoded, but instead while processing, enforcing that we do not allocate
	 * more memory.
	 * 
	 * @param m          The input matrix that is not delta encoded and should not be modified
	 * @param transposed If the input matrix is transposed.
	 * @param rowCols    The list of columns to encode
	 * @return A delta encoded encoding.
	 */
	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, int[] rowCols) {
		final int sampleSize = transposed ? m.getNumColumns() : m.getNumRows();
		return createFromMatrixBlockDelta(m, transposed, rowCols, sampleSize);
	}

	/**
	 * Encode a delta representation of the matrix input taking the first "sampleSize" rows into account.
	 * 
	 * Note the input matrix should not be delta encoded, but instead while processing, enforcing that we do not allocate
	 * more memory.
	 * 
	 * @param m          Input matrix that is not delta encoded and should not be modified
	 * @param transposed If the input matrix is transposed.
	 * @param rowCols    The list of columns to encode
	 * @param sampleSize The number of rows to consider for the delta encoding (from the beginning)
	 * @return A delta encoded encoding.
	 */
	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, int[] rowCols, int sampleSize) {
		throw new NotImplementedException();
	}

	/**
	 * Create encoding of a single specific column inside the matrix input.
	 * 
	 * @param m          The Matrix to encode a column from
	 * @param transposed If the matrix is in transposed format.
	 * @param rowCol     The column index to encode
	 * @return An encoded format of the information of this column.
	 */
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

	private static IEncode createFromDenseTransposed(MatrixBlock m, int row) {
		final DenseBlock db = m.getDenseBlock();
		if(!db.isContiguous())
			throw new NotImplementedException("Not Implemented non contiguous dense matrix encoding for sample");
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
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

		if(map.getOrDefault(0, -1) > nCol / 4) {
			map.replaceWithUIDsNoZero();
			final int zeroCount = map.get(0);
			final int nV = nCol - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique - 1);

			// for(int i = off, r = 0, di = 0; i < end; i += nCol, r++){
			for(int i = off, r = 0, di = 0; i < end; i++, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.get(vals[i]));
				}
			}

			final AOffset o = OffsetFactory.createOffset(offsets);
			return new SparseEncoding(d, o, zeroCount, nCol);
		}
		else {
			map.replaceWithUIDs();
			// Create output map
			final AMapToData d = MapToFactory.create(nCol, nUnique);

			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i++, r++)
				d.set(r, map.get(vals[i]));

			return new DenseEncoding(d);
		}
	}

	private static IEncode createFromSparseTransposed(MatrixBlock m, int row) {
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final SparseBlock sb = m.getSparseBlock();
		if(sb.isEmpty(row))
			return new EmptyEncoding();
		final int apos = sb.pos(row);
		final int alen = sb.size(row) + apos;
		final double[] avals = sb.values(row);
		final int[] aix = sb.indexes(row);

		// Iteration 1 of non zero values, make Count HashMap.
		for(int i = apos; i < alen; i++) // sequential of non zero cells.
			map.increment(avals[i]);

		final int nUnique = map.size();

		map.replaceWithUIDs();

		final int nCol = m.getNumColumns();
		if(alen - apos > nCol / 4) { // return a dense encoding
			// If the row was full but the overall matrix is sparse.
			final int correct = (alen - apos == m.getNumColumns()) ? 0 : 1;
			final AMapToData d = MapToFactory.create(nCol, nUnique + correct);
			// Since the dictionary is allocated with zero then we exploit that here and
			// only iterate through non zero entries.
			for(int i = apos; i < alen; i++)
				// correction one to assign unique IDs taking into account zero
				d.set(aix[i], map.get(avals[i]) + correct);
			// the rest is automatically set to zero.

			return new DenseEncoding(d);
		}
		else { // return a sparse encoding
			// Create output map
			final AMapToData d = MapToFactory.create(alen - apos, nUnique);

			// Iteration 2 of non zero values, make either a IEncode Dense or sparse map.
			for(int i = apos, j = 0; i < alen; i++, j++)
				d.set(j, map.get(avals[i]));

			// Iteration 3 of non zero indexes, make a Offset Encoding to know what cells are zero and not.
			// not done yet
			final AOffset o = OffsetFactory.createOffset(aix, apos, alen);
			final int zero = m.getNumColumns() - o.getSize();
			return new SparseEncoding(d, o, zero, m.getNumColumns());
		}
	}

	private static IEncode createFromDense(MatrixBlock m, int col) {
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

		if(map.getOrDefault(0, -1) > nRow / 4) {
			map.replaceWithUIDsNoZero();
			final int zeroCount = map.get(0);
			final int nV = m.getNumRows() - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique - 1);

			for(int i = off, r = 0, di = 0; i < end; i += nCol, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.get(vals[i]));
				}
			}

			final AOffset o = OffsetFactory.createOffset(offsets);

			return new SparseEncoding(d, o, zeroCount, nRow);
		}
		else {
			// Allocate counts, and iterate once to replace counts with u ids
			map.replaceWithUIDs();
			final AMapToData d = MapToFactory.create(nRow, nUnique);
			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i += nCol, r++)
				d.set(r, map.get(vals[i]));
			return new DenseEncoding(d);
		}
	}

	private static IEncode createFromSparse(MatrixBlock m, int col) {

		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final SparseBlock sb = m.getSparseBlock();

		final double guessedNumberOfNonZero = Math.min(4, Math.ceil((double) m.getNumRows() * m.getSparsity()));
		final IntArrayList offsets = new IntArrayList((int) guessedNumberOfNonZero);

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
		map.replaceWithUIDs();

		final AMapToData d = MapToFactory.create(offsets.size(), nUnique);

		// Iteration 2 of non zero values, make either a IEncode Dense or sparse map.
		for(int off = 0, r = 0; off < offsets.size(); r++) {
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

		final int zero = m.getNumRows() - offsets.size();
		return new SparseEncoding(d, o, zero, m.getNumRows());
	}

	private static IEncode createWithReader(MatrixBlock m, int[] rowCols, boolean transposed) {
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
		else if(map.size() == 1 && offsets.size() == nRows)
			return new ConstEncoding(nRows);

		map.replaceWithUIDs();
		if(offsets.size() < nRows / 4) {
			// Output encoded sparse since there is very empty.
			final int zeros = nRows - offsets.size();
			return createWithReaderSparse(m, map, zeros, rowCols, offsets, nRows, transposed);
		}
		else
			return createWithReaderDense(m, map, rowCols, nRows, transposed, offsets.size() < nRows);

	}

	private static IEncode createWithReaderDense(MatrixBlock m, DblArrayCountHashMap map, int[] rowCols, int nRows,
		boolean transposed, boolean zero) {
		// Iteration 2,
		final int unique = map.size() + (zero ? 1 : 0);
		final ReaderColumnSelection reader2 = ReaderColumnSelection.createReader(m, rowCols, transposed);
		final AMapToData d = MapToFactory.create(nRows, unique);

		DblArray cellVals;
		if(zero)
			while((cellVals = reader2.nextRow()) != null)
				d.set(reader2.getCurrentRowIndex(), map.get(cellVals) + 1);
		else
			while((cellVals = reader2.nextRow()) != null)
				d.set(reader2.getCurrentRowIndex(), map.get(cellVals));

		return new DenseEncoding(d);
	}

	private static IEncode createWithReaderSparse(MatrixBlock m, DblArrayCountHashMap map, int zeros, int[] rowCols,
		IntArrayList offsets, int nRows, boolean transposed) {
		final ReaderColumnSelection reader2 = ReaderColumnSelection.createReader(m, rowCols, transposed);
		DblArray cellVals = reader2.nextRow();

		final AMapToData d = MapToFactory.create(offsets.size(), map.size());

		int i = 0;
		// Iterator 2 of non zero tuples.
		while(cellVals != null) {
			d.set(i++, map.get(cellVals));
			cellVals = reader2.nextRow();
		}

		final AOffset o = OffsetFactory.createOffset(offsets);

		return new SparseEncoding(d, o, zeros, nRows);
	}
}
