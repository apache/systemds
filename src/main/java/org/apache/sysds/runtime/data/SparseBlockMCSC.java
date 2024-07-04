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

package org.apache.sysds.runtime.data;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

import org.apache.sysds.utils.MemoryEstimates;

/**
 * SparseBlock implementation that realizes a 'modified compressed sparse column' representation, where each compressed
 * column is stored as a separate SparseRow object which provides flexibility for unsorted column appends without the
 * need for global reshifting of values/indexes but it incurs additional memory overhead per column for object/array
 * headers per column which also slows down memory-bound operations due to higher memory bandwidth requirements.
 *
 * TODO implement row interface of sparse blocks (can be slow but must be correct;
 * additionally, we can expose the column API for efficient use in specific operations)
 */

public class SparseBlockMCSC extends SparseBlock {

	private static final long serialVersionUID = 112364695245614881L;

	private SparseRow[] _columns = null;
	private int _clenInferred = -1;

	public SparseBlockMCSC(SparseBlock sblock, int clen) {
		_clenInferred = clen;
		initialize(sblock);
	}

	public SparseBlockMCSC(SparseBlock sblock) {
		initialize(sblock);
	}

	private void initialize(SparseBlock sblock) {
		int clen = 0;

		if(sblock instanceof SparseBlockMCSC) {
			SparseRow[] originalColumns = ((SparseBlockMCSC) sblock)._columns;
			_columns = new SparseRow[originalColumns.length];
			for(int i = 0; i < _columns.length; i++) {
				if(originalColumns[i] != null)
					_columns[i] = originalColumns[i].copy(true);
			}
		}
		else if(sblock instanceof SparseBlockMCSR) {
			SparseRow[] originalRows = ((SparseBlockMCSR) sblock).getRows();
			Map<Integer, Integer> columnSizes = new HashMap<>();
			if(_clenInferred == -1) {
				for(SparseRow row : originalRows) {
					if(row != null && !row.isEmpty()) {
						for(int i = 0; i < row.size(); i++) {
							int rowIndex = row.indexes()[i];
							columnSizes.put(rowIndex, columnSizes.getOrDefault(rowIndex, 0) + 1);
						}
					}
				}
				clen = columnSizes.keySet().stream().max(Integer::compare).orElseThrow(NoSuchElementException::new);
				_columns = new SparseRow[clen + 1];
			}
			else {
				_columns = new SparseRow[_clenInferred];
			}

			for(int i = 0; i < _columns.length; i++) {
				int columnSize = columnSizes.getOrDefault(i, -1);
				if(columnSize == -1) {
					continue;
				}
				else if(columnSize == 1) {
					_columns[i] = new SparseRowScalar();
				}
				else { //columnSize > 1
					_columns[i] = new SparseRowVector(columnSize);
				}
			}

			int[] rowIndexes = null;
			double[] values = null;
			int rowPosition = 0;
			for(SparseRow row : originalRows) {
				if(row != null && !row.isEmpty()) {
					rowIndexes = row.indexes();
					values = row.values();
					for(int i = 0; i < row.size(); i++) {
						int rowIndex = rowIndexes[i];
						double currentValue = values[i];
						_columns[rowIndex].set(rowPosition, currentValue);
					}
				}
				rowPosition++;
			}

		}
		// general case SparseBlock
		else {
			HashMap<Integer, Integer> columnSizes = new HashMap<>();
			int[] columnIndexes = sblock.indexes(0);
			for(int col : columnIndexes) {
				columnSizes.put(col, columnSizes.getOrDefault(col, 0) + 1);
			}

			clen = columnSizes.keySet().stream().max(Integer::compare).orElseThrow(NoSuchElementException::new);
			if(_clenInferred == -1)
				_columns = new SparseRow[clen + 1];
			else
				_columns = new SparseRow[_clenInferred];
			for(int i = 0; i < _columns.length; i++) {
				int columnSize = columnSizes.getOrDefault(i, -1);
				if(columnSize == -1) {
					continue;
				}
				else if(columnSize == 1) {
					_columns[i] = new SparseRowScalar();
				}
				else { //columnSize > 1
					_columns[i] = new SparseRowVector(columnSize);
				}
			}

			double[] vals = sblock.values(0);
			int[] cols = sblock.indexes(0);
			int row = 0;
			int i = 0;
			while(i < vals.length) {
				int rowSize = sblock.size(row);
				for(int j = i; j < i + rowSize; j++) {
					_columns[cols[j]].set(row, vals[j]);
				}
				i += rowSize;
				row++;
			}
		}
	}

	public SparseBlockMCSC(SparseRow[] cols, boolean deep) {
		if(deep) {
			_columns = new SparseRow[cols.length];
			for(int i = 0; i < _columns.length; i++) {
				_columns[i] = (cols[i].size() == 1) ? new SparseRowScalar(cols[i].indexes()[0],
					cols[i].values()[0]) : new SparseRowVector(cols[i]);
			}
		}
		else {
			_columns = cols;
		}
	}

	public SparseBlockMCSC(int clen) {
		_columns = new SparseRow[clen];
	}

	public SparseBlockMCSC(int rlen, int clen) {
		this(clen);
	}

	/**
	 * Get the estimated in-memory size of the sparse block in MCSC with the given dimensions w/o accounting for
	 * overallocation.
	 *
	 * @param nrows    number of rows
	 * @param ncols    number of columns
	 * @param sparsity sparsity ratio
	 * @return memory estimate
	 */
	public static long estimateSizeInMemory(long nrows, long ncols, double sparsity) {
		double nnz = Math.ceil(sparsity * nrows * ncols);
		double clen = Math.min(nrows, nnz); // num sparse column objects
		double rnnz = Math.max(SparseRowVector.initialCapacity, nnz / clen);

		// Each sparse column has a fixed overhead of 16B (object) + 12B (3 ints),
		// 24B (int array), 24B (double array), i.e., in total 76B
		// Each non-zero value requires 12B for the row-index/value pair.
		// Overheads for arrays, objects, and references refer to 64bit JVMs
		// If nnz < columns we have guaranteed also empty columns.
		double size = 16; //object
		size += MemoryEstimates.objectArrayCost(ncols); //references
		long sparseColSize = 16; // object
		sparseColSize += 2 * 4; // 2 integers + padding
		sparseColSize += MemoryEstimates.intArrayCost(0);
		sparseColSize += MemoryEstimates.doubleArrayCost(0);
		sparseColSize += 12 * Math.max(1, rnnz); //avoid bias by down cast for ultra-sparse
		size += clen * sparseColSize; //sparse columns

		// robustness for long overflows
		return (long) Math.min(size, Long.MAX_VALUE);
	}

	/**
	 * Computes the exact size in memory of the materialized block
	 *
	 * @return the exact size in memory
	 */
	public long getExactSizeInMemory() {
		double size = 16; //object
		size += MemoryEstimates.objectArrayCost(_columns.length); //references

		for(SparseRow sc : _columns) {
			if(sc == null)
				continue;
			long sparseColSize = 16; // object
			if(sc instanceof SparseRowScalar) {
				sparseColSize += 12;
			}
			else { //SparseRowVector
				sparseColSize += 2 * 4; // 2 integers
				sparseColSize += MemoryEstimates.intArrayCost(0);
				sparseColSize += MemoryEstimates.doubleArrayCost(0);
				sparseColSize += 12 * ((SparseRowVector) sc).capacity();
			}
			size += sparseColSize; //sparse columns
		}

		// robustness for long overflows
		return (long) Math.min(size, Long.MAX_VALUE);
	}

	///////////////////
	//SparseBlock implementation

	@Override
	public void allocate(int c) {
		if(!isAllocated(c)) {
			_columns[c] = new SparseRowVector();
		}
	}

	@Override
	public void allocate(int c, int nnz) {
		if(!isAllocated(c)) {
			_columns[c] = (nnz == 1) ? new SparseRowScalar() : new SparseRowVector(nnz);
		}
	}

	@Override
	public void allocate(int c, int ennz, int maxnnz) {
		if(!isAllocated(c)) {
			_columns[c] = (ennz == 1) ? new SparseRowScalar() : new SparseRowVector(ennz, maxnnz);
		}
	}

	@Override
	public void compact(int c) {
		if(isAllocated(c)) {
			if(_columns[c] instanceof SparseRowVector && _columns[c].size() > SparseBlock.INIT_CAPACITY &&
				_columns[c].size() * SparseBlock.RESIZE_FACTOR1 < ((SparseRowVector) _columns[c]).capacity()) {
				((SparseRowVector) _columns[c]).compact();
			}
			else if(_columns[c] instanceof SparseRowScalar) {
				SparseRowScalar s = (SparseRowScalar) _columns[c];
				if(s.getValue() == 0)
					_columns[c] = null;
			}
		}

	}

	@Override
	public int numRows() {
		// this is a column-oriented layout
		return 0;
	}

	public int numCols() {
		return _columns.length;
	}

	@Override
	public boolean isThreadSafe() {
		return true;
	}

	@Override
	public boolean isContiguous() {
		return false;
	}

	@Override
	public boolean isAllocated(int c) {
		return _columns[c] != null;
	}

	@Override
	public void reset() {
		for(SparseRow col : _columns) {
			if(col != null) {
				col.reset(col.size(), Integer.MAX_VALUE);
			}
		}
	}

	@Override
	public void reset(int ennz, int maxnnz) {
		for(SparseRow col : _columns) {
			if(col != null) {
				col.reset(ennz, maxnnz);
			}
		}
	}

	@Override
	public void reset(int c, int ennz, int maxnnz) {
		if(isAllocated(c)) {
			_columns[c].reset(ennz, maxnnz);
		}
	}

	@Override
	public long size() {
		long nnz = 0;
		for(SparseRow col : _columns) {
			if(col != null) {
				nnz += col.size();
			}
		}
		return nnz;
	}

	@Override
	public int size(int c) {
		//prior check with isEmpty(r) expected
		return isAllocated(c) ? _columns[c].size() : 0;
	}

	@Override
	public long size(int cl, int cu) {
		long nnz = 0;
		for(int i = cl; i < cu; i++) {
			nnz += isAllocated(i) ? _columns[i].size() : 0;
		}
		return nnz;
	}

	@Override
	public long size(int rl, int ru, int cl, int cu) {
		long nnz = 0;
		for(int i = cl; i < cu; i++) {
			if(!isEmpty(i)) {
				int start = posFIndexGTE(rl, i);
				int end = posFIndexGTE(ru, i);
				nnz += (start != -1) ? (end - start) : 0;
			}
		}
		return nnz;
	}

	@Override
	public boolean isEmpty(int c) {
		return _columns[c] == null || _columns[c].isEmpty();
	}

	@Override
	public boolean checkValidity(int rlen, int clen, long nnz, boolean strict) {
		//1. Correct meta data
		if(rlen < 0 || clen < 0)
			throw new RuntimeException("Invalid block dimensions: (" + rlen + ", " + clen + ").");

		//2. Correct array lengths
		if(size() < nnz)
			throw new RuntimeException("Incorrect size: " + size() + " (expected: " + nnz + ").");

		//3. Sorted column indices per row
		for(int i = 0; i < clen; i++) {
			if(isEmpty(i))
				continue;
			int apos = pos(i);
			int alen = size(i);
			int[] aix = indexes(i);
			double[] avals = values(i);
			for(int k = apos + 1; k < apos + alen; k++) {
				if(aix[k - 1] >= aix[k] | aix[k - 1] < 0) {
					throw new RuntimeException(
						"Wrong sparse column ordering, at column=" + i + ", pos=" + k + " with row indexes " +
							aix[k - 1] + ">=" + aix[k]);
				}
				if(avals[k] == 0) {
					throw new RuntimeException(
						"The values are expected to be non zeros " + "but zero at column: " + i + ", row pos: " + k);
				}
			}
		}
		//4. A capacity that is no larger than nnz times resize factor
		for(int i = 0; i < clen; i++) {
			long max_size = (long) Math.max(nnz * RESIZE_FACTOR1, INIT_CAPACITY);
			if(!isEmpty(i) && values(i).length > max_size) {
				throw new RuntimeException(
					"The capacity is larger than nnz times a resize factor(=2). " + "Actual length = " +
						values(i).length + ", should not exceed " + max_size);
			}
		}

		return true;
	}

	@Override
	public int[] indexes(int c) {
		//prior check with isEmpty(c) expected
		return _columns[c].indexes();
	}

	@Override
	public double[] values(int c) {
		//prior check with isEmpty(c) expected
		return _columns[c].values();
	}

	@Override
	public int pos(int c) {
		//arrays per column (always start 0)
		return 0;
	}

	@Override
	public boolean set(int r, int c, double v) {
		if(!isAllocated(c)) {
			_columns[c] = new SparseRowScalar();
		}
		else if(_columns[c] instanceof SparseRowScalar && !_columns[c].isEmpty()) {
			_columns[c] = new SparseRowVector(_columns[c]);
		}
		return _columns[c].set(r, v);
	}

	@Override
	public void set(int c, SparseRow col, boolean deep) {
		//copy values into existing column to avoid allocation
		if(isAllocated(c) && _columns[c] instanceof SparseRowVector &&
			((SparseRowVector) _columns[c]).capacity() >= col.size() && deep) {
			((SparseRowVector) _columns[c]).copy(col);
			//set new sparse column (incl allocation if required)
		}
		else {
			_columns[c] = (deep && col != null) ? new SparseRowVector(col) : col;
		}
	}

	@Override
	public boolean add(int r, int c, double v) {
		if(!isAllocated(c)) {
			_columns[c] = new SparseRowScalar();
		}
		else if(_columns[c] instanceof SparseRowScalar && !_columns[c].isEmpty()) {
			SparseRowScalar s = (SparseRowScalar) _columns[c];
			if(s.getIndex() == r) {
				return s.set(s.getIndex(), v + s.getValue());
			}
			else {
				_columns[c] = new SparseRowVector(_columns[c]);
			}
		}
		return _columns[c].add(r, v);
	}

	@Override
	public void append(int r, int c, double v) {
		if(v == 0) {
			return;
		}
		else if(_columns[c] == null) {
			_columns[c] = new SparseRowScalar(r, v);
		}
		else {
			_columns[c] = _columns[c].append(r, v);
		}
	}

	@Override
	public void setIndexRange(int c, int rl, int ru, double[] v, int vix, int vlen) {
		if(!isAllocated(c)) {
			_columns[c] = new SparseRowVector();
		}
		else if(_columns[c] instanceof SparseRowScalar) {
			_columns[c] = new SparseRowVector(_columns[c]);
		}
		((SparseRowVector) _columns[c]).setIndexRange(rl, ru - 1, v, vix, vlen);
	}

	@Override
	public void setIndexRange(int c, int rl, int ru, double[] v, int[] vix, int vpos, int vlen) {
		if(!isAllocated(c)) {
			_columns[c] = new SparseRowVector();
		}
		else if(_columns[c] instanceof SparseRowScalar) {
			_columns[c] = new SparseRowVector(_columns[c]);
		}
		//different sparse row semantics: upper bound inclusive
		((SparseRowVector) _columns[c]).setIndexRange(rl, ru - 1, v, vix, vpos, vlen);
	}

	@Override
	public void deleteIndexRange(int c, int rl, int ru) {
		//prior check with isEmpty(c) expected
		//different sparse row semantics: upper bound inclusive
		if(_columns[c] instanceof SparseRowScalar) {
			_columns[c] = new SparseRowVector(_columns[c]);
		}
		((SparseRowVector) _columns[c]).deleteIndexRange(rl, ru - 1);
	}

	@Override
	public void sort() {
		for(SparseRow col : _columns) {
			if(col != null && !col.isEmpty()) {
				col.sort();
			}
		}
	}

	@Override
	public void sort(int c) {
		//prior check with isEmpty(c) expected
		_columns[c].sort();
	}

	@Override
	public double get(int r, int c) {
		if(!isAllocated(c)) {
			return 0;
		}
		return _columns[c].get(r);
	}

	@Override
	public SparseRow get(int c) {
		return _columns[c];
	}

	@Override
	public int posFIndexLTE(int r, int c) {
		//prior check with isEmpty(c) expected
		if(_columns[c] instanceof SparseRowScalar) {
			_columns[c] = new SparseRowVector(_columns[c]);
		}
		return ((SparseRowVector) _columns[c]).searchIndexesFirstLTE(r);
	}

	@Override
	public int posFIndexGTE(int r, int c) {
		return _columns[c].searchIndexesFirstGTE(r);
	}

	@Override
	public int posFIndexGT(int r, int c) {
		return _columns[c].searchIndexesFirstGT(r);
	}

	@Override
	public Iterator<Integer> getNonEmptyRowsIterator(int rl, int ru) {
		throw new UnsupportedOperationException("Non-empty rows iterator is not supported in column layouts.");
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		final int nCol = numCols();
		sb.append("SparseBlockMCSC: clen=");
		sb.append(nCol);
		sb.append(", nnz=");
		sb.append(size());
		sb.append("\n");
		final int colDigits = (int) Math.max(Math.ceil(Math.log10(nCol)), 1);
		for(int i = 0; i < nCol; i++) {
			if(isEmpty(i))
				continue;
			sb.append(String.format("%0" + colDigits + "d %s\n", i, _columns[i].toString()));
		}

		return sb.toString();
	}

	public SparseRow[] getCols() {
		return _columns;
	}
}
