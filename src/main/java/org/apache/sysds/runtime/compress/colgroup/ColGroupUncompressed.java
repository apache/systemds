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
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.SortUtils;

/**
 * Column group type for columns that are stored as dense arrays of doubles. Uses a MatrixBlock internally to store the
 * column contents.
 * 
 */
public class ColGroupUncompressed extends AColGroup {
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
	 * @param transposed     Says if the input matrix raw block have been transposed. This should not ever be true since
	 *                       we still have the original matrixBlock in case of aborting the compression.
	 */
	public ColGroupUncompressed(int[] colIndicesList, MatrixBlock rawBlock, boolean transposed) {
		super(colIndicesList, transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows());

		// prepare meta data
		int numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		// Create a matrix with just the requested rows of the original block
		_data = new MatrixBlock(numRows, _colIndexes.length, rawBlock.isInSparseFormat());

		// ensure sorted col indices
		if(!SortUtils.isSorted(0, _colIndexes.length, _colIndexes))
			Arrays.sort(_colIndexes);

		// special cases empty blocks
		if(rawBlock.isEmptyBlock(false))
			return;

		// special cases full block
		if(!transposed && _data.getNumColumns() == rawBlock.getNumColumns()) {
			_data.copy(rawBlock);
			return;
		}

		// dense implementation for dense and sparse matrices to avoid linear search
		int m = numRows;
		int n = _colIndexes.length;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				double val = transposed ? rawBlock.quickGetValue(_colIndexes[j], i) : rawBlock.quickGetValue(i,
					_colIndexes[j]);
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
	protected ColGroupUncompressed(List<AColGroup> groupsToDecompress) {
		super(mergeColIndices(groupsToDecompress), groupsToDecompress.get(0)._numRows);

		// Invert the list of column indices
		int maxColIndex = _colIndexes[_colIndexes.length - 1];
		int[] colIndicesInverted = new int[maxColIndex + 1];
		for(int i = 0; i < _colIndexes.length; i++) {
			colIndicesInverted[_colIndexes[i]] = i;
		}

		// Create the buffer that holds the uncompressed data, packed together
		_data = new MatrixBlock(_numRows, _colIndexes.length, false);

		for(AColGroup colGroup : groupsToDecompress) {
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
	public ColGroupType getColGroupType() {
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
	private static int[] mergeColIndices(List<AColGroup> groupsToDecompress) {
		// Pass 1: Determine number of columns
		int sz = 0;
		for(AColGroup colGroup : groupsToDecompress) {
			sz += colGroup.getNumCols();
		}

		// Pass 2: Copy column offsets out
		int[] ret = new int[sz];
		int pos = 0;
		for(AColGroup colGroup : groupsToDecompress) {
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
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false))
			return;
		for(int row = rl; row < ru; row++, offT++) {
			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(offT, col, target.quickGetValue(offT, col) + cellVal);
			}
		}
	}

	private void decompressToBlockUncompressed(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false))
			return;
		for(int row = rl; row < ru; row++, offT++) {
			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(offT, col, target.quickGetValue(offT, col) + cellVal);
			}
		}
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUncompressed(target, rl, ru, offT, values);
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUncompressed(target, rl, ru, offT, values);
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
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
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
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false)) {
			return;
		}
		// Run through the rows, putting values into the appropriate locations
		for(int row = rl; row < ru; row++) {
			double cellVal = _data.quickGetValue(row, colpos);
			// Apparently rows are cols here.
			target.quickSetValue(0, row, cellVal);
		}
	}

	@Override
	public void decompressColumnToBlock(double[] target, int colpos, int rl, int ru) {
		// empty block, nothing to add to output
		if(_data.isEmptyBlock(false)) {
			return;
		}
		// Run through the rows, putting values into the appropriate locations
		for(int row = rl; row < ru; row++) {
			double cellVal = _data.quickGetValue(row, colpos);
			// Apparently rows are cols here.
			target[row] += cellVal;
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
		LibMatrixMult.matrixMult(_data, shortVector, result, rl, ru);
	}

	public void rightMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		// Pull out the relevant rows of the vector

		int clen = _colIndexes.length;
		MatrixBlock subMatrix = new MatrixBlock(clen, matrix.getNumColumns(), false);
		subMatrix.allocateDenseBlock();
		double[] b = subMatrix.getDenseBlockValues();

		for(int colIx = 0; colIx < clen; colIx++) {
			int row = _colIndexes[colIx];
			for(int col = 0; col < matrix.getNumColumns(); col++)
				b[colIx * matrix.getNumColumns() + col] = matrix.quickGetValue(row, col);
		}

		subMatrix.setNonZeros(clen * matrix.getNumColumns());

		// // Multiply the selected columns by the appropriate parts of the vector
		LibMatrixMult.matrixMult(_data, subMatrix, result);
	}

	public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
		int ru) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] c) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] c, int numVals, double[] values) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultByMatrix(double[] vector, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
		int vOff) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row,
		double[] MaterializedRow) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
	}

	public double computeMxx(double c, Builtin builtin) {
		throw new NotImplementedException("Not implemented max min on uncompressed");
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
	public AColGroup scalarOperation(ScalarOperator op) {
		// execute scalar operations
		MatrixBlock retContent = _data.scalarOperations(op, new MatrixBlock());
		// construct new uncompressed column group
		return new ColGroupUncompressed(getColIndices(), _data.getNumRows(), retContent);
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		throw new NotImplementedException("Should not be called use other matrix function for uncompressed columns");
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
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\n");
		sb.append(_data.getNumColumns() + " ");
		sb.append(_data.getNumRows() + " ");
		sb.append(_data.getNonZeros() + " ");
		sb.append(_data.isInSparseFormat() + " ");
		// sb.append(_data.toString());
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

	@Override
	public AColGroup sliceColumns(int cl, int cu) {
		throw new NotImplementedException("Not implemented slice columns");
	}

	public double getMin() {
		return _data.min();
	}

	public double getMax() {
		return _data.max();
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] result, int offT) {
		throw new NotImplementedException("Not implemented slice columns");
	}

	@Override
	public void leftMultByRowVector(double[] vector, double[] result, int numVals, double[] values, int offT) {
		throw new NotImplementedException("Not implemented slice columns");

	}

	@Override
	public void leftMultBySelfDiagonalColGroup(double[] result, int numColumns) {
		throw new NotImplementedException("Not implemented slice columns");
	}

	@Override
	public AColGroup copy() {
		throw new NotImplementedException("Not implemented copy of uncompressed colGroup yet.");
	}

	@Override
	public boolean containsValue(double pattern){
		return _data.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(){
		return _data.getNonZeros();
	}
}
