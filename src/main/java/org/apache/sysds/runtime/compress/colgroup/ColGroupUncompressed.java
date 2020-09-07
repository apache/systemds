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

package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.SortUtils;

/**
 * Column group type for columns that are stored as dense arrays of doubles. Uses a MatrixBlock internally to store the
 * column contents.
 * 
 */
public class ColGroupUncompressed extends ColGroup {
	private static final long serialVersionUID = 4870546053280378891L;

	/**
	 * We store the contents of the columns as a MatrixBlock to take advantage of high-performance routines available
	 * for this data structure.
	 */
	private MatrixBlock _data;

	protected ColGroupUncompressed() {
		super();
	}

	public long getValuesSize() {
		throw new DMLCompressionException("Should not currently be used to estimate uncompressed size.");
	}

	/**
	 * Main constructor for Uncompressed ColGroup.
	 * 
	 * @param colIndicesList Indices (relative to the current block) of the columns that this column group represents.
	 * @param rawBlock       The uncompressed block; uncompressed data must be present at the time that the constructor
	 *                       is called
	 * @param compSettings   The Settings for how to compress this block, Here using information about the raw block if
	 *                       it is transposed.
	 */
	protected ColGroupUncompressed(int[] colIndicesList, MatrixBlock rawBlock, CompressionSettings compSettings) {
		super(colIndicesList, compSettings.transposeInput ? rawBlock.getNumColumns() : rawBlock.getNumRows());

		// prepare meta data
		int numRows = compSettings.transposeInput ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		// Create a matrix with just the requested rows of the original block
		_data = new MatrixBlock(numRows, _colIndexes.length, rawBlock.isInSparseFormat());

		// ensure sorted col indices
		if(!SortUtils.isSorted(0, _colIndexes.length, _colIndexes))
			Arrays.sort(_colIndexes);

		// special cases empty blocks
		if(rawBlock.isEmptyBlock(false))
			return;
		// special cases full block
		if(!compSettings.transposeInput && _data.getNumColumns() == rawBlock.getNumColumns()) {
			_data.copy(rawBlock);
			return;
		}

		// dense implementation for dense and sparse matrices to avoid linear search
		int m = numRows;
		int n = _colIndexes.length;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				double val = compSettings.transposeInput ? rawBlock.quickGetValue(_colIndexes[j], i) : rawBlock
					.quickGetValue(i, _colIndexes[j]);
				_data.appendValue(i, j, val);
			}
		}
		_data.examSparsity();

		// convert sparse MCSR to read-optimized CSR representation
		if(_data.isInSparseFormat()) {
			_data = new MatrixBlock(_data, Type.CSR, false);
		}
	}

	/**
	 * Constructor for creating temporary decompressed versions of one or more compressed column groups.
	 * 
	 * @param groupsToDecompress compressed columns to subsume. Must contain at least one element.
	 */
	protected ColGroupUncompressed(List<ColGroup> groupsToDecompress) {
		super(mergeColIndices(groupsToDecompress), groupsToDecompress.get(0)._numRows);

		// Invert the list of column indices
		int maxColIndex = _colIndexes[_colIndexes.length - 1];
		int[] colIndicesInverted = new int[maxColIndex + 1];
		for(int i = 0; i < _colIndexes.length; i++) {
			colIndicesInverted[_colIndexes[i]] = i;
		}

		// Create the buffer that holds the uncompressed data, packed together
		_data = new MatrixBlock(_numRows, _colIndexes.length, false);

		for(ColGroup colGroup : groupsToDecompress) {
			colGroup.decompressToBlock(_data, colIndicesInverted);
		}
	}

	/**
	 * Constructor for internal use. Used when a method needs to build an instance of this class from scratch.
	 * 
	 * @param colIndices column mapping for this column group
	 * @param numRows    number of rows in the column, for passing to the superclass
	 * @param data       matrix block
	 */
	protected ColGroupUncompressed(int[] colIndices, int numRows, MatrixBlock data) {
		super(colIndices, numRows);
		_data = data;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.UNCOMPRESSED;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.UNCOMPRESSED;
	}

	/**
	 * Access for superclass
	 * 
	 * @return direct pointer to the internal representation of the columns
	 */
	public MatrixBlock getData() {
		return _data;
	}

	/**
	 * Subroutine of constructor.
	 * 
	 * @param groupsToDecompress input to the constructor that decompresses into a temporary UncompressedColGroup
	 * @return a merged set of column indices across all those groups
	 */
	private static int[] mergeColIndices(List<ColGroup> groupsToDecompress) {
		// Pass 1: Determine number of columns
		int sz = 0;
		for(ColGroup colGroup : groupsToDecompress) {
			sz += colGroup.getNumCols();
		}

		// Pass 2: Copy column offsets out
		int[] ret = new int[sz];
		int pos = 0;
		for(ColGroup colGroup : groupsToDecompress) {
			int[] tmp = colGroup.getColIndices();
			System.arraycopy(tmp, 0, ret, pos, tmp.length);
			pos += tmp.length;
		}

		// Pass 3: Sort and return the list of columns
		Arrays.sort(ret);
		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeUncompressed(_numRows, getNumCols(), _data.getSparsity());
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false))
			return;
		for(int row = rl; row < ru; row++) {
			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(row, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false)) {
			return;
		}
		// Run through the rows, putting values into the appropriate locations
		for(int row = 0; row < _data.getNumRows(); row++) {
			for(int colIx = 0; colIx < _data.getNumColumns(); colIx++) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(row, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false)) {
			return;
		}
		// Run through the rows, putting values into the appropriate locations
		for(int row = 0; row < _data.getNumRows(); row++) {
			double cellVal = _data.quickGetValue(row, colpos);
			// Apparently rows are cols here.
			target.quickSetValue(0, row, cellVal);
		}
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in uncompressed group.");

		// uncompressed get value
		return _data.quickGetValue(r, ix);
	}

	@Override
	public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
		throw new NotImplementedException("Should not be called use other matrix function");
	}

	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru) {
		// Pull out the relevant rows of the vector
		int clen = _colIndexes.length;

		MatrixBlock shortVector = new MatrixBlock(clen, 1, false);
		shortVector.allocateDenseBlock();
		double[] b = shortVector.getDenseBlockValues();
		for(int colIx = 0; colIx < clen; colIx++)
			b[colIx] = vector.quickGetValue(_colIndexes[colIx], 0);
		shortVector.recomputeNonZeros();

		// Multiply the selected columns by the appropriate parts of the vector
		LibMatrixMult.matrixMult(_data, shortVector, result,rl,ru);
	}

	public void rightMultByMatrix(MatrixBlock vector, MatrixBlock result, int rl, int ru) {
		// Pull out the relevant rows of the vector
		// int clen = _colIndexes.length;

		// MatrixBlock subMatrix = new MatrixBlock(clen, vector.getNumColumns(), false);
		// subMatrix.allocateDenseBlock();
		// double[] b = subMatrix.getDenseBlockValues();
		// TODO Fix, to copy correctly
		throw new NotImplementedException("Dense right block uncompressed column multiplication not implemented yet.");
		// for(int colIx = 0; colIx < clen; colIx++)
		// 	b[colIx] = vector.quickGetValue(_colIndexes[colIx], 0);
		// subMatrix.recomputeNonZeros();

		// // Multiply the selected columns by the appropriate parts of the vector
		// LibMatrixMult.matrixMult(_data, subMatrix, result);
	}

	@Override
	public void rightMultByMatrix(double[] matrix, double[] result, int numVals, double[] values, int rl, int ru, int vOff){
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] c, int numVals) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] c, int numVals, double[] values) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByMatrix(double[] vector, double[] c, int numVals, double[] values, int numRows, int numCols, int rl, int ru, int vOff) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}


	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result) {
		MatrixBlock pret = new MatrixBlock(matrix.getNumRows(), _colIndexes.length, false);
		LibMatrixMult.matrixMult(matrix, _data, pret);

		// copying partialResult to the proper indices of the result
		if(!pret.isEmptyBlock(false)) {
			double[] rsltArr = result.getDenseBlockValues();
			for(int colIx = 0; colIx < _colIndexes.length; colIx++)
				rsltArr[_colIndexes[colIx]] = pret.quickGetValue(0, colIx);
			result.recomputeNonZeros();
		}
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		// execute scalar operations
		MatrixBlock retContent = _data.scalarOperations(op, new MatrixBlock());
		// construct new uncompressed column group
		return new ColGroupUncompressed(getColIndices(), _data.getNumRows(), retContent);
	}

	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] ret) {
		throw new NotImplementedException("Should not be called");
	}

	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock ret) {
		// execute unary aggregate operations
		LibMatrixAgg.aggregateUnaryMatrix(_data, ret, op);
		ret = ret.allocateBlock();
		// shift result into correct column indexes
		if(op.indexFn instanceof ReduceRow) {
			// shift partial results, incl corrections
			for(int i = _colIndexes.length - 1; i >= 0; i--) {
				double val = ret.quickGetValue(0, i);
				ret.quickSetValue(0, i, 0);
				ret.quickSetValue(0, _colIndexes[i], val);
				if(op.aggOp.existsCorrection())
					for(int j = 1; j < ret.getNumRows(); j++) {
						double corr = ret.quickGetValue(j, i);
						ret.quickSetValue(j, i, 0);
						ret.quickSetValue(j, _colIndexes[i], corr);
					}
			}
		}
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] result, int rl, int ru) {
		throw new NotImplementedException("Unimplemented Specific Sub ColGroup Aggregation Operation");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		// read col contents (w/ meta data)
		_data = new MatrixBlock();
		_data.readFields(in);
		_numRows = _data.getNumRows();

		// read col indices
		int numCols = _data.getNumColumns();
		_colIndexes = new int[numCols];
		for(int i = 0; i < numCols; i++)
			_colIndexes[i] = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// write col contents first (w/ meta data)
		_data.write(out);

		// write col indices
		int len = _data.getNumColumns();
		for(int i = 0; i < len; i++)
			out.writeInt(_colIndexes[i]);
	}

	@Override
	public long getExactSizeOnDisk() {
		return _data.getExactSizeOnDisk() + 4 * _data.getNumColumns();
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		for(int i = rl; i < ru; i++)
			rnnz[i - rl] += _data.recomputeNonZeros(i, i, 0, _data.getNumColumns() - 1);
	}

	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		// UC iterator is always row major, so no need for custom handling
		return new UCIterator(rl, ru, inclZeros);
	}

	@Override
	public ColGroupRowIterator getRowIterator(int rl, int ru) {
		return new UCRowIterator(rl, ru);
	}

	private class UCIterator implements Iterator<IJV> {
		// iterator configuration
		private final int _ru;
		private final boolean _inclZeros;

		// iterator state
		private final IJV _buff = new IJV();
		private int _rpos = -1;
		private int _cpos = -1;
		private double _value = 0;

		public UCIterator(int rl, int ru, boolean inclZeros) {
			_ru = ru;
			_inclZeros = inclZeros;
			_rpos = rl;
			_cpos = -1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public IJV next() {
			_buff.set(_rpos, _colIndexes[_cpos], _value);
			getNextValue();
			return _buff;
		}

		private void getNextValue() {
			do {
				boolean nextRow = (_cpos + 1 >= getNumCols());
				_rpos += nextRow ? 1 : 0;
				_cpos = nextRow ? 0 : _cpos + 1;
				if(_rpos >= _ru)
					return; // reached end
				_value = _data.quickGetValue(_rpos, _cpos);
			}
			while(!_inclZeros && _value == 0);
		}
	}

	private class UCRowIterator extends ColGroupRowIterator {
		public UCRowIterator(int rl, int ru) {
			// do nothing
		}

		@Override
		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			// copy entire dense/sparse row
			if(_data.isAllocated()) {
				if(_data.isInSparseFormat()) {
					if(!_data.getSparseBlock().isEmpty(rowIx)) {
						SparseBlock sblock = _data.getSparseBlock();
						int apos = sblock.pos(rowIx);
						int alen = sblock.size(rowIx);
						int[] aix = sblock.indexes(rowIx);
						double[] avals = sblock.values(rowIx);
						for(int k = apos; k < apos + alen; k++)
							buff[_colIndexes[aix[k]]] = avals[k];
					}
				}
				else {
					final int clen = getNumCols();
					double[] a = _data.getDenseBlockValues();
					for(int j = 0, aix = rowIx * clen; j < clen; j++)
						buff[_colIndexes[j]] = a[aix + j];
				}
			}
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\n");
		sb.append(_data.toString());
		return sb.toString();
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		return _data;
	}

	@Override
	public boolean getIfCountsType() {
		return false;
	}

	@Override
	public double[] getValues() {
		if(_data.isInSparseFormat()) {
			return _data.getSparseBlock().values(0);
		}
		else {
			return _data.getDenseBlock().values(0);
		}
	}

	@Override
	public boolean isLossy() {
		return false;
	}

}
