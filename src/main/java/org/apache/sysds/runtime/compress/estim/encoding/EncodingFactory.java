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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
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

	public static final Log LOG = LogFactory.getLog(EncodingFactory.class.getName());

	/**
	 * Encode a list of columns together from the input matrix, as if it is cocoded.
	 * 
	 * @param m          The matrix input to encode
	 * @param transposed If the matrix is transposed in memory
	 * @param rowCols    The list of columns to encode.
	 * @return An encoded format of the information of the columns.
	 */
	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, IColIndex rowCols) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(rowCols.size() == 1) {
			return createFromMatrixBlock(m, transposed, rowCols.get(0), null);
		}
		else {
			return createWithReader(m, rowCols, transposed, null);
		}
	}

	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, IColIndex rowCols,
		double[] scaleFactors) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(rowCols.size() == 1) {
			return createFromMatrixBlock(m, transposed, rowCols.get(0), scaleFactors);
		}
		else {
			return createWithReader(m, rowCols, transposed, scaleFactors);
		}
	}

	/**
	 * Encode a full delta representation of the matrix input taking all rows into account.
	 * 
	 * Note the input matrix should not be delta encoded, but instead while processing, enforcing that we do not
	 * allocate more memory.
	 * 
	 * @param m          The input matrix that is not delta encoded and should not be modified
	 * @param transposed If the input matrix is transposed.
	 * @param rowCols    The list of columns to encode
	 * @return A delta encoded encoding.
	 */
	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, IColIndex rowCols) {
		throw new NotImplementedException();
		// final int sampleSize = transposed ? m.getNumColumns() : m.getNumRows();
		// return createFromMatrixBlockDelta(m, transposed, rowCols, sampleSize);
	}

	/**
	 * Encode a delta representation of the matrix input taking the first "sampleSize" rows into account.
	 * 
	 * Note the input matrix should not be delta encoded, but instead while processing, enforcing that we do not
	 * allocate more memory.
	 * 
	 * @param m          Input matrix that is not delta encoded and should not be modified
	 * @param transposed If the input matrix is transposed.
	 * @param rowCols    The list of columns to encode
	 * @param sampleSize The number of rows to consider for the delta encoding (from the beginning)
	 * @return A delta encoded encoding.
	 */
	public static IEncode createFromMatrixBlockDelta(MatrixBlock m, boolean transposed, IColIndex rowCols,
		int sampleSize) {
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
		else {
			return createFromDense(m, rowCol);
		}
	}

	/**
	 * Create encoding of a single specific column inside the matrix input.
	 * 
	 * @param m            The Matrix to encode a column from
	 * @param transposed   If the matrix is in transposed format.
	 * @param rowCol       The column index to encode
	 * @param scaleFactors For quantization-fused compression, scale factors per row, or a single value for entire
	 *                     matrix
	 * @return An encoded format of the information of this column.
	 */
	public static IEncode createFromMatrixBlock(MatrixBlock m, boolean transposed, int rowCol, double[] scaleFactors) {
		if(m.isEmpty())
			return new EmptyEncoding();
		else if(transposed) {
			if(scaleFactors != null) {
				if(m.isInSparseFormat())
					throw new NotImplementedException();
				else
					return createFromDenseTransposedQuantized(m, rowCol, scaleFactors); 
			}
			else {
				if(m.isInSparseFormat())
					return createFromSparseTransposed(m, rowCol);
				else
					return createFromDenseTransposed(m, rowCol);
			}
		}
		else if(m.isInSparseFormat()) {
			if(scaleFactors != null) {
				throw new NotImplementedException(); // TODO: handle quantization-fused compression
			}
			else {
				return createFromSparse(m, rowCol);
			}
		}
		else {
			if(scaleFactors != null) {
				return createFromDenseQuantized(m, rowCol, scaleFactors);
			}
			else {
				return createFromDense(m, rowCol);
			}
		}
	}

	public static IEncode create(ColGroupConst c) {
		return new ConstEncoding(-1);
	}

	public static IEncode create(ColGroupEmpty c) {
		return new EmptyEncoding();
	}

	public static IEncode create(AMapToData d) {
		return new DenseEncoding(d);
	}

	public static IEncode create(AMapToData d, AOffset i, int nRow) {
		return new SparseEncoding(d, i, nRow);
	}

	private static IEncode createFromDenseTransposed(MatrixBlock m, int row) {
		final DenseBlock db = m.getDenseBlock();
		if(!db.isContiguous())
			throw new NotImplementedException("Not Implemented non contiguous dense matrix encoding for sample");
		final DoubleCountHashMap map = new DoubleCountHashMap();
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
		else if(nUnique == 0)
			return new EmptyEncoding();
		else if(map.getOrDefault(0.0, -1) > nCol / 4) {
			map.replaceWithUIDsNoZero();
			final int zeroCount = map.get(0.0);
			final int nV = nCol - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique - 1);
			int di = 0;
			for(int i = off, r = 0; i < end; i++, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.getId(vals[i]));
				}
			}
			if(di != nV)
				throw new RuntimeException("Did not find equal number of elements " + di + " vs " + nV);

			final AOffset o = OffsetFactory.createOffset(offsets);
			return new SparseEncoding(d, o, nCol);
		}
		else {
			// Create output map
			final AMapToData d = MapToFactory.create(nCol, nUnique);

			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i++, r++)
				d.set(r, map.getId(vals[i]));

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
		for(int i = apos; i < alen; i++) {
			map.increment(avals[i]);
		}

		final int nUnique = map.size();

		final int nCol = m.getNumColumns();
		if(nUnique == 0)
			return new EmptyEncoding();
		else if(alen - apos > nCol / 4) { // return a dense encoding
			// If the row was full but the overall matrix is sparse.
			final int correct = (alen - apos == m.getNumColumns()) ? 0 : 1;
			final AMapToData d = MapToFactory.create(nCol, nUnique + correct);
			// Since the dictionary is allocated with zero then we exploit that here and
			// only iterate through non zero entries.
			for(int i = apos; i < alen; i++)
				d.set(aix[i], map.getId(avals[i]) + correct);

			// the rest is automatically set to zero.
			return new DenseEncoding(d);
		}
		else { // return a sparse encoding
				// Create output map
			final AMapToData d = MapToFactory.create(alen - apos, nUnique);

			// Iteration 2 of non zero values, make either a IEncode Dense or sparse map.
			for(int i = apos, j = 0; i < alen; i++, j++)
				d.set(j, map.getId(avals[i]));

			// Iteration 3 of non zero indexes, make a Offset Encoding to know what cells are zero and not.
			// not done yet
			try {

				final AOffset o = OffsetFactory.createOffset(aix, apos, alen);
				return new SparseEncoding(d, o, m.getNumColumns());
			}
			catch(Exception e) {
				String mes = Arrays.toString(Arrays.copyOfRange(aix, apos, alen)) + "\n" + apos + "  " + alen;
				mes += Arrays.toString(Arrays.copyOfRange(avals, apos, alen));
				throw new DMLRuntimeException(mes, e);
			}
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

		if(map.getOrDefault(0.0, -1) > nRow / 4) {
			map.replaceWithUIDsNoZero();
			final int zeroCount = map.get(0.0);
			final int nV = m.getNumRows() - zeroCount;
			final IntArrayList offsets = new IntArrayList(nV);

			final AMapToData d = MapToFactory.create(nV, nUnique - 1);
			int di = 0;
			for(int i = off, r = 0; i < end; i += nCol, r++) {
				if(vals[i] != 0) {
					offsets.appendValue(r);
					d.set(di++, map.getId(vals[i]));
				}
			}
			if(di != nV)
				throw new DMLRuntimeException("Invalid number of zero.");

			final AOffset o = OffsetFactory.createOffset(offsets);

			return new SparseEncoding(d, o, nRow);
		}
		else {
			// Allocate counts, and iterate once to replace counts with u ids

			final AMapToData d = MapToFactory.create(nRow, nUnique);
			// Iteration 2, make final map
			for(int i = off, r = 0; i < end; i += nCol, r++)
				d.set(r, map.getId(vals[i]));
			return new DenseEncoding(d);
		}
	}

	private static IEncode createFromSparse(MatrixBlock m, int col) {

		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final SparseBlock sb = m.getSparseBlock();

		final double guessedNumberOfNonZero = Math.min(4, Math.ceil(m.getNumRows() * m.getSparsity()));
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
			if(index >= 0) {
				final double v = sb.values(r)[index];
				if(index >= 0)
					d.set(off++, map.getId(v));
			}
		}

		// Iteration 3 of non zero indexes, make a Offset Encoding to know what cells are zero and not.
		final AOffset o = OffsetFactory.createOffset(offsets);
		return new SparseEncoding(d, o, m.getNumRows());
	}

	private static IEncode createFromDenseTransposedQuantized(MatrixBlock m, int row, double[] scaleFactors) {
		final DenseBlock db = m.getDenseBlock();
		if(!db.isContiguous())
			throw new NotImplementedException("Not Implemented non contiguous dense matrix encoding for sample");
		final DoubleCountHashMap map = new DoubleCountHashMap();
		final int off = db.pos(row);
		final int nCol = m.getNumColumns();
		final int end = off + nCol;
		final double[] vals = db.values(row);

		// Validate scaleFactors
		boolean useSingleScalar = false;
		if(scaleFactors != null) {
			if(scaleFactors.length == 1) {
				useSingleScalar = true;
			}
		}

		if(useSingleScalar == true) {

			// Iteration 1: Apply scaling & quantization, then populate the HashMap
			for(int i = off; i < end; i++) // sequential access
				map.increment(Math.floor(vals[i] * scaleFactors[0]));

			final int nUnique = map.size();

			if(nUnique == 1)
				return new ConstEncoding(m.getNumColumns());
			else if(nUnique == 0)
				return new EmptyEncoding();
			else if(map.getOrDefault(0.0, -1) > nCol / 4) {
				map.replaceWithUIDsNoZero();
				final int zeroCount = map.get(0.0);
				final int nV = nCol - zeroCount;
				final IntArrayList offsets = new IntArrayList(nV);

				final AMapToData d = MapToFactory.create(nV, nUnique - 1);
				int di = 0;
				for(int i = off, r = 0; i < end; i++, r++) {
					double value = Math.floor(vals[i] * scaleFactors[0]);
					if (value != 0) {
						offsets.appendValue(r);
						d.set(di++, map.getId(value));
					}
				}
				if(di != nV)
					throw new RuntimeException("Did not find equal number of elements " + di + " vs " + nV);

				final AOffset o = OffsetFactory.createOffset(offsets);
				return new SparseEncoding(d, o, nCol);
			}
			else {
				// Create output map
				final AMapToData d = MapToFactory.create(nCol, nUnique);

				// Iteration 2, make final map
				for(int i = off, r = 0; i < end; i++, r++) {
					double value = Math.floor(vals[i] * scaleFactors[0]);
					d.set(r, map.getId(value));
				}

				return new DenseEncoding(d);
			}
		}
		else {
			// Iteration 1: Apply scaling & quantization, then populate the HashMap
			for(int i = off; i < end; i++) // sequential access
				map.increment(Math.floor(vals[i] * scaleFactors[row]));

			final int nUnique = map.size();

			if(nUnique == 1)
				return new ConstEncoding(m.getNumColumns());
			else if(nUnique == 0)
				return new EmptyEncoding();
			else if(map.getOrDefault(0.0, -1) > nCol / 4) {
				map.replaceWithUIDsNoZero();
				final int zeroCount = map.get(0.0);
				final int nV = nCol - zeroCount;
				final IntArrayList offsets = new IntArrayList(nV);

				final AMapToData d = MapToFactory.create(nV, nUnique - 1);
				int di = 0;
				for(int i = off, r = 0; i < end; i++, r++) {
					double value = Math.floor(vals[i] * scaleFactors[row]);
					if (value != 0) {
						offsets.appendValue(r);
						d.set(di++, map.getId(value));
					}
				}
				if(di != nV)
					throw new RuntimeException("Did not find equal number of elements " + di + " vs " + nV);

				final AOffset o = OffsetFactory.createOffset(offsets);
				return new SparseEncoding(d, o, nCol);
			}
			else {
				// Create output map
				final AMapToData d = MapToFactory.create(nCol, nUnique);

				// Iteration 2, make final map
				for(int i = off, r = 0; i < end; i++, r++) {
					double value = Math.floor(vals[i] * scaleFactors[row]);
					d.set(r, map.getId(value));
				}

				return new DenseEncoding(d);
			}

		}
	}

	private static IEncode createFromDenseQuantized(MatrixBlock m, int col, double[] scaleFactors) {
		final DenseBlock db = m.getDenseBlock();
		if(!db.isContiguous())
			throw new NotImplementedException("Not Implemented non contiguous dense matrix encoding for sample");
		final DoubleCountHashMap map = new DoubleCountHashMap(16);
		final int off = col;
		final int nCol = m.getNumColumns();
		final int nRow = m.getNumRows();
		final int end = off + nRow * nCol;
		final double[] vals = m.getDenseBlockValues();

		// Validate scaleFactors
		boolean useSingleScalar = false;
		if(scaleFactors != null) {
			if(scaleFactors.length == 1) {
				useSingleScalar = true;
			}
		}

		if(useSingleScalar == true) {
			// Iteration 1, make Count HashMap with quantized values
			for(int i = off; i < end; i += nCol) {// jump down through rows.
				map.increment(Math.floor(vals[i] * scaleFactors[0]));
			}
			final int nUnique = map.size();
			if(nUnique == 1)
				return new ConstEncoding(m.getNumColumns());

			if(map.getOrDefault(0.0, -1) > nRow / 4) {
				map.replaceWithUIDsNoZero();
				final int zeroCount = map.get(0.0);
				final int nV = m.getNumRows() - zeroCount;
				final IntArrayList offsets = new IntArrayList(nV);

				final AMapToData d = MapToFactory.create(nV, nUnique - 1);
				int di = 0;
				for(int i = off, r = 0; i < end; i += nCol, r++) {
					double value = Math.floor(vals[i] * scaleFactors[0]);
					if(value != 0) {
						offsets.appendValue(r);
						d.set(di++, map.getId(value));
					}
				}
				if(di != nV)
					throw new DMLRuntimeException("Invalid number of zero.");

				final AOffset o = OffsetFactory.createOffset(offsets);

				return new SparseEncoding(d, o, nRow);
			}
			else {
				// Allocate counts, and iterate once to replace counts with u ids

				final AMapToData d = MapToFactory.create(nRow, nUnique);
				// Iteration 2, make final map with quantized values
				for(int i = off, r = 0; i < end; i += nCol, r++) {
					double value = Math.floor(vals[i] * scaleFactors[0]);
					d.set(r, map.getId(value));
				}
				return new DenseEncoding(d);
			}
		}
		else {
			// Iteration 1, make Count HashMap with row-wise quantized values
			for(int i = off, r = 0; i < end; i += nCol, r++) {// jump down through rows.
				map.increment(Math.floor(vals[i] * scaleFactors[r]));
			}
			final int nUnique = map.size();
			if(nUnique == 1)
				return new ConstEncoding(m.getNumColumns());

			if(map.getOrDefault(0.0, -1) > nRow / 4) {
				map.replaceWithUIDsNoZero();
				final int zeroCount = map.get(0.0);
				final int nV = m.getNumRows() - zeroCount;
				final IntArrayList offsets = new IntArrayList(nV);

				final AMapToData d = MapToFactory.create(nV, nUnique - 1);
				int di = 0;
				for(int i = off, r = 0; i < end; i += nCol, r++) {
					double value = Math.floor(vals[i] * scaleFactors[r]);
					if(value != 0) {
						offsets.appendValue(r);
						d.set(di++, map.getId(value));
					}
				}
				if(di != nV)
					throw new DMLRuntimeException("Invalid number of zero.");

				final AOffset o = OffsetFactory.createOffset(offsets);

				return new SparseEncoding(d, o, nRow);
			}
			else {
				// Allocate counts, and iterate once to replace counts with u ids

				final AMapToData d = MapToFactory.create(nRow, nUnique);
				// Iteration 2, make final map with row-wise quantized values
				for(int i = off, r = 0; i < end; i += nCol, r++) {
					double value = Math.floor(vals[i] * scaleFactors[r]);
					d.set(r, map.getId(value));
				}
				return new DenseEncoding(d);
			}
		}
	}

	private static IEncode createWithReader(MatrixBlock m, IColIndex rowCols, boolean transposed,
		double[] scaleFactors) {
		final ReaderColumnSelection reader1 = (scaleFactors == null) ? ReaderColumnSelection.createReader(m, rowCols,
			transposed) : ReaderColumnSelection.createQuantizedReader(m, rowCols, transposed, scaleFactors);
		final int nRows = transposed ? m.getNumColumns() : m.getNumRows();
		final DblArrayCountHashMap map = new DblArrayCountHashMap();
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

		if(offsets.size() < nRows / 4)
			// Output encoded sparse since there is very empty.
			return createWithReaderSparse(m, map, rowCols, offsets, nRows, transposed, scaleFactors);
		else
			return createWithReaderDense(m, map, rowCols, nRows, transposed, offsets.size() < nRows, scaleFactors);

	}

	private static IEncode createWithReaderDense(MatrixBlock m, DblArrayCountHashMap map, IColIndex rowCols, int nRows,
		boolean transposed, boolean zero, double[] scaleFactors) {
		// Iteration 2,
		final int unique = map.size() + (zero ? 1 : 0);
		final ReaderColumnSelection reader2 = (scaleFactors == null) ? ReaderColumnSelection.createReader(m, rowCols,
			transposed) : ReaderColumnSelection.createQuantizedReader(m, rowCols, transposed, scaleFactors);
		final AMapToData d = MapToFactory.create(nRows, unique);

		DblArray cellVals;
		if(zero)
			while((cellVals = reader2.nextRow()) != null)
				d.set(reader2.getCurrentRowIndex(), map.getId(cellVals) + 1);
		else
			while((cellVals = reader2.nextRow()) != null)
				d.set(reader2.getCurrentRowIndex(), map.getId(cellVals));

		return new DenseEncoding(d);
	}

	private static IEncode createWithReaderSparse(MatrixBlock m, DblArrayCountHashMap map, IColIndex rowCols,
		IntArrayList offsets, int nRows, boolean transposed, double[] scaleFactors) {
		final ReaderColumnSelection reader2 = (scaleFactors == null) ? ReaderColumnSelection.createReader(m, rowCols,
			transposed) : ReaderColumnSelection.createQuantizedReader(m, rowCols, transposed, scaleFactors);
		DblArray cellVals = reader2.nextRow();

		final AMapToData d = MapToFactory.create(offsets.size(), map.size());

		int i = 0;
		// Iterator 2 of non zero tuples.
		while(cellVals != null) {
			d.set(i++, map.getId(cellVals));
			cellVals = reader2.nextRow();
		}

		final AOffset o = OffsetFactory.createOffset(offsets);

		return new SparseEncoding(d, o, nRows);
	}

	public static SparseEncoding createSparse(AMapToData map, AOffset off, int nRows) {
		return new SparseEncoding(map, off, nRows);
	}
}
