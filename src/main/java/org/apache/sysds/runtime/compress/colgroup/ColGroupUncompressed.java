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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
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

	/**
	 * Constructor for serialization
	 */
	protected ColGroupUncompressed() {
		super();
	}

	/**
	 * Main constructor for Uncompressed ColGroup.
	 * 
	 * @param colIndicesList Indices (relative to the current block) of the columns that this column group represents.
	 * @param rawBlock       The uncompressed block; uncompressed data must be present at the time that the constructor
	 *                       is called
	 * @param transposed     Says if the input matrix raw block have been transposed.
	 */
	public ColGroupUncompressed(int[] colIndicesList, MatrixBlock rawBlock, boolean transposed) {
		super(colIndicesList);
		final int _numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		if(colIndicesList.length == 1) {
			final int col = colIndicesList[0];
			if(transposed) {
				_data = rawBlock.slice(col, col, 0, rawBlock.getNumColumns() - 1);
				_data = LibMatrixReorg.transposeInPlace(_data, 1);
			}
			else
				_data = rawBlock.slice(0, rawBlock.getNumRows() - 1, col, col);

			return;
		}

		if(rawBlock.isInSparseFormat() && transposed) {
			_data = new MatrixBlock();
			_data.setNumRows(_numRows);
			_data.setNumColumns(colIndicesList.length);
		}

		// Create a matrix with just the requested rows of the original block
		_data = new MatrixBlock(_numRows, _colIndexes.length, rawBlock.isInSparseFormat());

		// ensure sorted col indices
		if(!SortUtils.isSorted(0, _colIndexes.length, _colIndexes))
			Arrays.sort(_colIndexes);

		// special cases empty blocks
		if(rawBlock.isEmptyBlock(false))
			return;

		// special cases full blocks
		if(!transposed && _data.getNumColumns() == rawBlock.getNumColumns()) {
			_data.copy(rawBlock);
			return;
		}

		// dense implementation for dense and sparse matrices to avoid linear search
		int m = _numRows;
		int n = _colIndexes.length;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				double val = transposed ? rawBlock.quickGetValue(_colIndexes[j], i) : rawBlock.quickGetValue(i,
					_colIndexes[j]);
				_data.appendValue(i, j, val);
			}
		}
		_data.examSparsity();
	}

	/**
	 * Constructor for internal use. Used when a method needs to build an instance of this class from scratch.
	 * 
	 * @param colIndices column mapping for this column group
	 * @param data       matrix block
	 */
	protected ColGroupUncompressed(int[] colIndices, MatrixBlock data) {
		super(colIndices);
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

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeUncompressed(_data.getNumRows(), getNumCols(), _data.getSparsity());
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		double[] c = target.getDenseBlockValues();
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		long nnz = 0;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int row = rl; row < ru; row++, offT += tCol) {
				if(!sb.isEmpty(row)) {
					int apos = sb.pos(row);
					int alen = sb.size(row) + apos;
					int[] aix = sb.indexes(row);
					double[] avals = sb.values(row);
					nnz += alen;
					for(int col = apos; col < alen; col++) {
						c[_colIndexes[aix[col]] + offT] += avals[col];
					}
				}
			}
		}
		else {
			values = _data.getDenseBlockValues();
			offT = offT * tCol;
			int offS = rl * nCol;
			for(int row = rl; row < ru; row++, offT += tCol, offS += nCol) {
				for(int j = 0; j < nCol; j++) {
					final double v = values[offS + j];
					if(v != 0) {
						c[offT + _colIndexes[j]] += v;
						nnz++;
					}
				}
			}
		}
		target.setNonZeros(nnz + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		double[] c = target.getDenseBlockValues();
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			if(sb == null)
				return;
			for(int row = rl; row < ru; row++, offT += tCol) {
				if(!sb.isEmpty(row)) {
					int apos = sb.pos(row);
					int alen = sb.size(row) + apos;
					int[] aix = sb.indexes(row);
					double[] avals = sb.values(row);
					for(int col = apos; col < alen; col++)
						c[_colIndexes[aix[col]] + offT] += avals[col];

				}
			}
		}
		else {
			values = _data.getDenseBlockValues();
			offT = offT * tCol;
			int offS = rl * nCol;
			for(int row = rl; row < ru; row++, offT += tCol, offS += nCol)
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offS + j];

		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		double[] c = target.getDenseBlockValues();
		int nnz = 0;
		int off = colpos;
		if(_data.isInSparseFormat()) {
			for(int i = 0; i < _data.getNumRows(); i++) {
				c[i] += _data.quickGetValue(i, colpos);
				if(c[i] != 0)
					nnz++;
			}
		}
		else {
			double[] denseValues = _data.getDenseBlockValues();
			for(int i = 0; i < _data.getNumRows(); i++, off += _colIndexes.length) {
				c[i] += denseValues[off];
				if(c[i] != 0)
					nnz++;
			}
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void decompressColumnToBlock(double[] target, int colpos, int rl, int ru) {
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public double get(int r, int c) {
		final int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			return 0;
		else
			return _data.quickGetValue(r, ix);
	}

	// @Override
	// public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
	// throw new NotImplementedException("Should not be called use other matrix function");
	// }

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

	public void leftMultByMatrix(MatrixBlock matrix, double[] result, int numCols, int rl, int ru) {

		MatrixBlock tmpRet = new MatrixBlock(ru - rl, _data.getNumColumns(), false);
		tmpRet.allocateDenseBlock();
		MatrixBlock leftSlice = matrix.slice(rl, ru - 1, false);
		LibMatrixMult.matrixMult(leftSlice, _data, tmpRet);
		int offT = numCols * rl;

		if(tmpRet.isEmpty())
			return;
		if(tmpRet.isInSparseFormat()) {
			SparseBlock sb = tmpRet.getSparseBlock();
			for(int rowIdx = 0; rowIdx < ru - rl; rowIdx++, offT += numCols) {
				if(!sb.isEmpty(rowIdx)) {
					final int apos = sb.pos(rowIdx);
					final int alen = sb.size(rowIdx) + apos;
					final int[] aix = sb.indexes(rowIdx);
					final double[] avals = sb.values(rowIdx);
					for(int col = apos; col < alen; col++)
						result[offT + _colIndexes[aix[col]]] += avals[col];
				}
			}
		}
		else {
			double[] tmpRetV = tmpRet.getDenseBlockValues();
			for(int j = rl, offTemp = 0; j < ru; j++, offTemp += _colIndexes.length, offT += numCols) {
				for(int i = 0; i < _colIndexes.length; i++)
					result[offT + _colIndexes[i]] += tmpRetV[offTemp + i];
			}
		}
	}

	public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int rl,
		int ru) {
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
		if(retContent.isEmpty())
			return new ColGroupEmpty(_colIndexes, _data.getNumRows());
		// construct new uncompressed column group
		return new ColGroupUncompressed(getColIndices(), retContent);
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		DenseBlock b = new DenseBlockFP64(new int[] {1, v.length}, v);
		MatrixBlock that = new MatrixBlock(1, v.length, b);
		that.setNonZeros(v.length);
		MatrixBlock resultBlock = new MatrixBlock();
		if(left)
			that.binaryOperations(op, _data, resultBlock);
		else
			_data.binaryOperations(op, that, resultBlock);
		return new ColGroupUncompressed(_colIndexes, resultBlock, false);
	}

	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] ret) {
		unaryAggregateOperations(op, ret, 0, _data.getNumRows());
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] result, int rl, int ru) {
		LOG.warn("Inefficient Unary Aggregate because of Uncompressed ColumnGroup");
		// Since usually Uncompressed column groups are used in case of extreme sparsity, it is fine
		// using a slice, since we dont allocate extra just extract the pointers to the sparse rows.
		MatrixBlock tmpData = _data.slice(rl, ru - 1, false);
		MatrixBlock tmp = tmpData.aggregateUnaryOperations(op, new MatrixBlock(), _data.getNumRows(),
			new MatrixIndexes(1, 1), true);
		if(tmp.isEmpty()) {
			if(op.aggOp.increOp.fn instanceof Builtin) {
				Builtin b = (Builtin) op.aggOp.increOp.fn;
				if(op.indexFn instanceof ReduceRow)
					for(int i = 0; i < _colIndexes.length; i++)
						result[_colIndexes[i]] = b.execute(result[_colIndexes[i]], 0);
				else if(op.indexFn instanceof ReduceAll)
					result[0] = b.execute(result[0], 0);
				else
					for(int row = rl; row < ru; row++)
						result[row] = b.execute(result[row], 0);
			}
			return;
		}

		tmp.sparseToDense();
		// The output is always dense in unary aggregates.
		double[] tmpV = tmp.getDenseBlockValues();

		if(op.aggOp.increOp.fn instanceof Builtin) {
			Builtin b = (Builtin) op.aggOp.increOp.fn;
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes[i]] = b.execute(result[_colIndexes[i]], tmpV[i]);
			else if(op.indexFn instanceof ReduceAll)
				result[0] = b.execute(result[0], tmpV[0]);
			else
				for(int i = 0, row = rl; i < tmpV.length; i++, row++)
					result[row] = b.execute(result[row], tmpV[i]);
		}
		else {
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes[i]] += tmpV[i];
			else if(op.indexFn instanceof ReduceAll)
				result[0] += tmpV[0];
			else
				for(int i = 0, row = rl; i < tmpV.length; i++, row++)
					result[row] += tmpV[i];
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		// read col contents (w/ meta data)
		_data = new MatrixBlock();
		_data.readFields(in);

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
		sb.append(" numCols : " + _data.getNumColumns());
		sb.append(" numRows : " + _data.getNumRows());
		sb.append(" nonZeros: " + _data.getNonZeros());
		sb.append(" Sparse  : " + _data.isInSparseFormat());
		sb.append("\n");

		if(!_data.isInSparseFormat() && _data.getNumRows() < 100000)
			sb.append(Arrays.toString(_data.getDenseBlockValues()));
		else if(_data.getNumRows() < 100)
			sb.append(_data.toString());
		else
			sb.append(" dont print uncompressed matrix because it is to big.");

		return sb.toString();
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		return _data;
	}

	@Override
	public double[] getValues() {
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();

			if(sb == null || sb.isEmpty(0))
				return null;
			else
				return _data.getSparseBlock().values(0);
		}
		else
			return _data.getDenseBlock().values(0);
	}

	@Override
	public boolean isLossy() {
		return false;
	}

	@Override
	public double getMin() {
		return _data.min();
	}

	@Override
	public double getMax() {
		return _data.max();
	}

	@Override
	public void tsmm(double[] result, int numColumns) {
		MatrixBlock tmp = new MatrixBlock(_colIndexes.length, _colIndexes.length, true);
		LibMatrixMult.matrixMultTransposeSelf(_data, tmp, true, false);
		double[] tmpV = tmp.getDenseBlockValues();
		for(int i = 0, offD = 0, offT = 0; i < numColumns; i++, offD += numColumns, offT += _colIndexes.length)
			for(int j = i; j < numColumns; j++)
				result[offD + _colIndexes[j]] += tmpV[offT + j];

	}

	@Override
	public AColGroup copy() {
		throw new NotImplementedException("Not implemented copy of uncompressed colGroup yet.");
	}

	@Override
	public boolean containsValue(double pattern) {
		return _data.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros() {
		return _data.getNonZeros();
	}

	@Override
	public int getNumRows() {
		return _data.getNumRows();
	}

	@Override
	public boolean isDense() {
		return !_data.isInSparseFormat();
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, double[] result, final int numRows, final int numCols) {
		if(lhs instanceof ColGroupEmpty)
			return;
		if(lhs instanceof ColGroupUncompressed) {
			throw new DMLCompressionException("Not Implemented");
		}
		else {

			LOG.warn("Inefficient transpose of uncompressed to fit to"
				+ " t(AColGroup) %*% UncompressedColGroup mult by colGroup uncompressed column"
				+ " Currently solved by t(t(Uncompressed) %*% AColGroup");
			double[] tmpTransposedResult = new double[result.length];

			MatrixBlock ucCG = getData();
			MatrixBlock tmp = new MatrixBlock(ucCG.getNumColumns(), ucCG.getNumRows(), ucCG.isInSparseFormat());
			LibMatrixReorg.transpose(ucCG, tmp, InfrastructureAnalyzer.getLocalParallelism());
			lhs.leftMultByMatrix(tmp, tmpTransposedResult, numRows);

			for(int row = 0; row < numRows; row++) {
				for(int col = 0; col < numCols; col++) {
					result[row * numCols + col] += tmpTransposedResult[col * numRows + row];
				}
			}
		}
	}

	@Override
	protected AColGroup sliceSingleColumn(int col, int idx) {
		return sliceMultiColumns(idx, idx + 1, new int[] {0});
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		MatrixBlock newData = _data.slice(0, _data.getNumRows() - 1, idStart, idEnd - 1, true);
		if(newData.isEmpty())
			return new ColGroupEmpty(outputCols, newData.getNumRows());
		return new ColGroupUncompressed(outputCols, newData, false);
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		int[] outputCols = new int[right.getNumColumns()];
		for(int i = 0; i < outputCols.length; i++)
			outputCols[i] = i;
		MatrixBlock out = new MatrixBlock(_data.getNumRows(), right.getNumColumns(), true);
		LibMatrixMult.matrixMult(_data, right, out, InfrastructureAnalyzer.getLocalParallelism());
		return new ColGroupUncompressed(outputCols, out, false);
	}
}
