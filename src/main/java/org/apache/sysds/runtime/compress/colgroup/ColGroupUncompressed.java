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
import org.apache.sysds.runtime.compress.DMLCompressionException;
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
	private static final long serialVersionUID = -8254271148043292199L;
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
				_data = LibMatrixReorg.transposeInPlace(_data, InfrastructureAnalyzer.getLocalParallelism());
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
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT) {
		decompressToBlockUnSafe(target, rl, ru, offT);
		target.setNonZeros(_data.getNonZeros() + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT) {
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
			double[] values = _data.getDenseBlockValues();
			offT = offT * tCol;
			int offS = rl * nCol;
			for(int row = rl; row < ru; row++, offT += tCol, offS += nCol)
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[offS + j];

		}
	}

	@Override
	public double get(int r, int c) {
		final int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			return 0;
		else
			return _data.quickGetValue(r, ix);
	}

	@Override
	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {

		final MatrixBlock tmpRet = new MatrixBlock(ru - rl, _data.getNumColumns(), false);
		tmpRet.allocateDenseBlock();
		final MatrixBlock leftSlice = matrix.slice(rl, ru - 1, false);
		LibMatrixMult.matrixMult(leftSlice, _data, tmpRet);
		int offT = result.getNumColumns() * rl;
		final double[] resV = result.getDenseBlockValues();
		if(tmpRet.isEmpty())
			return;
		else if(tmpRet.isInSparseFormat()) {
			final SparseBlock sb = tmpRet.getSparseBlock();
			for(int rowIdx = 0; rowIdx < ru - rl; rowIdx++, offT += result.getNumColumns()) {
				if(sb.isEmpty(rowIdx))
					continue;

				final int apos = sb.pos(rowIdx);
				final int alen = sb.size(rowIdx) + apos;
				final int[] aix = sb.indexes(rowIdx);
				final double[] avals = sb.values(rowIdx);
				for(int col = apos; col < alen; col++)
					resV[offT + _colIndexes[aix[col]]] += avals[col];

			}
		}
		else {
			final double[] tmpRetV = tmpRet.getDenseBlockValues();
			for(int j = rl, offTemp = 0; j < ru; j++, offTemp += _colIndexes.length, offT += result.getNumColumns())
				for(int i = 0; i < _colIndexes.length; i++)
					resV[offT + _colIndexes[i]] += tmpRetV[offTemp + i];
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
		double[] selectedValues = new double[_colIndexes.length];
		for(int i = 0; i < _colIndexes.length; i++) {
			selectedValues[i] = v[_colIndexes[i]];
		}
		DenseBlock b = new DenseBlockFP64(new int[] {1, _colIndexes.length}, selectedValues);
		MatrixBlock that = new MatrixBlock(1, _colIndexes.length, b);
		that.setNonZeros(_colIndexes.length);
		MatrixBlock resultBlock = new MatrixBlock();

		if(left)
			that.binaryOperations(op, _data, resultBlock);
		else
			_data.binaryOperations(op, that, resultBlock);

		return new ColGroupUncompressed(_colIndexes, resultBlock);
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
	public final void tsmm(MatrixBlock ret) {
		double[] result = ret.getDenseBlockValues();
		int numColumns = ret.getNumColumns();
		tsmm(result, numColumns);
	}

	private void tsmm(double[] result, int numColumns) {
		final int tCol = _colIndexes.length;
		MatrixBlock tmp = new MatrixBlock(tCol, tCol, true);
		LibMatrixMult.matrixMultTransposeSelf(_data, tmp, true, false);
		if(tmp.getDenseBlock() == null && tmp.getSparseBlock() == null)
			return;
		else if(tmp.isInSparseFormat()) {
			throw new NotImplementedException("not Implemented sparse output of tsmm in compressed ColGroup.");
		}
		else {
			double[] tmpV = tmp.getDenseBlockValues();
			for(int row = 0, offTmp = 0; row < tCol; row++, offTmp += tCol) {
				final int offRet = _colIndexes[row] * numColumns;
				for(int col = row; col < tCol; col++)
					result[offRet + _colIndexes[col]] += tmpV[offTmp + col];
			}
		}
	}

	@Override
	public AColGroup copy() {
		MatrixBlock newData = new MatrixBlock(_data.getNumRows(), _data.getNumColumns(), _data.isInSparseFormat());
		// _data.copy(newData);
		newData.copy(_data);
		return new ColGroupUncompressed(_colIndexes, newData);
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
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		if(lhs instanceof ColGroupEmpty)
			return;
		if(lhs instanceof ColGroupUncompressed) {
			ColGroupUncompressed lhsUC = (ColGroupUncompressed) lhs;
			MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.length, 0);

			if(lhsUC._data == this._data) {

				LibMatrixMult.matrixMultTransposeSelf(this._data, tmpRet, true,
					InfrastructureAnalyzer.getLocalParallelism());
			}
			else {
				LOG.warn("Inefficient Left Matrix Multiplication with transpose of left hand side : t(l) %*% r");
				MatrixBlock lhData = lhsUC._data;
				MatrixBlock transposed = new MatrixBlock(lhData.getNumColumns(), lhData.getNumRows(), false);
				LibMatrixReorg.transpose(lhData, transposed, InfrastructureAnalyzer.getLocalParallelism());
				transposed.setNonZeros(lhData.getNonZeros());
				// do transposed left hand side, matrix multiplication.
				LibMatrixMult.matrixMult(transposed, this._data, tmpRet);
			}

			final double[] resV = result.getDenseBlockValues();
			if(tmpRet.isEmpty())
				return;
			else if(tmpRet.isInSparseFormat()) {
				SparseBlock sb = tmpRet.getSparseBlock();
				for(int row = 0; row < lhs._colIndexes.length; row++) {
					if(sb.isEmpty(row))
						continue;
					final int apos = sb.pos(row);
					final int alen = sb.size(row) + apos;
					final int[] aix = sb.indexes(row);
					final double[] avals = sb.values(row);
					final int offRes = lhs._colIndexes[row] * result.getNumColumns();
					for(int col = apos; col < alen; col++)
						resV[offRes + _colIndexes[aix[col]]] += avals[col];
				}
			}
			else {
				double[] tmpRetV = tmpRet.getDenseBlockValues();
				for(int row = 0; row < lhs._colIndexes.length; row++) {
					final int offRes = lhs._colIndexes[row] * result.getNumColumns();
					final int offTmp = lhs._colIndexes.length * row;
					for(int col = 0; col < _colIndexes.length; col++) {
						resV[offRes + _colIndexes[col]] += tmpRetV[offTmp + col];
					}
				}
			}
		}
		else {
			LOG.warn("\nInefficient transpose of uncompressed to fit to"
				+ " t(AColGroup) %*% UncompressedColGroup mult by colGroup uncompressed column"
				+ "\nCurrently solved by t(t(Uncompressed) %*% AColGroup)");
			MatrixBlock ucCG = getData();
			// make a function that allows the result of the mult to be directly output to a temporary matrix.
			MatrixBlock tmpTransposedResult = new MatrixBlock(ucCG.getNumColumns(), result.getNumColumns(), false);
			tmpTransposedResult.allocateDenseBlock();

			MatrixBlock tmp = new MatrixBlock(ucCG.getNumColumns(), ucCG.getNumRows(), ucCG.isInSparseFormat());
			LibMatrixReorg.transpose(ucCG, tmp, InfrastructureAnalyzer.getLocalParallelism());
			lhs.leftMultByMatrix(tmp, tmpTransposedResult);
			tmpTransposedResult.setNonZeros(ucCG.getNumColumns() * result.getNumColumns());

			final double[] resV = result.getDenseBlockValues();
			final int[] lhsC = lhs._colIndexes;
			final int[] rhsC = _colIndexes;

			// allocate the resulting matrix into the correct result indexes.
			// Note that the intermediate matrix is transposed, therefore the indexes are different than a normal
			// allocation.

			if(tmpTransposedResult.isEmpty())
				return;
			else if(tmpTransposedResult.isInSparseFormat())
				throw new NotImplementedException();
			else {
				final double[] tmpV = tmpTransposedResult.getDenseBlockValues();
				final int nCol = result.getNumColumns();

				for(int row = 0; row < rhsC.length; row++) {
					final int offR = rhsC[row];
					final int offT = row * nCol;
					for(int col = 0; col < lhsC.length; col++)
						resV[offR + lhsC[col] * nCol] += tmpV[offT + lhsC[col]];
				}
			}
		}
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		return sliceMultiColumns(idx, idx + 1, new int[] {0});
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		try {
			MatrixBlock newData = _data.slice(0, _data.getNumRows() - 1, idStart, idEnd - 1, true);
			if(newData.isEmpty())
				return new ColGroupEmpty(outputCols, newData.getNumRows());
			return new ColGroupUncompressed(outputCols, newData);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Error in slicing of uncompressed column group", e);
		}
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		final int nColR = right.getNumColumns();
		int[] outputCols = new int[nColR];
		for(int i = 0; i < outputCols.length; i++)
			outputCols[i] = i;
		if(_data.isEmpty() || right.isEmpty())
			return new ColGroupEmpty(outputCols, _data.getNumRows());
		MatrixBlock subBlockRight;

		if(right.isInSparseFormat()) {
			subBlockRight = new MatrixBlock(_data.getNumColumns(), nColR, true);
			subBlockRight.allocateSparseRowsBlock();
			final SparseBlock sbR = right.getSparseBlock();
			final SparseBlock subR = subBlockRight.getSparseBlock();
			for(int i = 0; i < _colIndexes.length; i++)
				subR.set(i, sbR.get(_colIndexes[i]), false);
		}
		else {
			subBlockRight = new MatrixBlock(_data.getNumColumns(), nColR, false);
			subBlockRight.allocateDenseBlock();
			final double[] sbr = subBlockRight.getDenseBlockValues();
			final double[] rightV = right.getDenseBlockValues();
			for(int i = 0; i < _colIndexes.length; i++) {
				final int offSubBlock = i * nColR;
				final int offRight = _colIndexes[i] * nColR;
				System.arraycopy(rightV, offRight, sbr, offSubBlock, nColR);
			}
		}
		// Hack to force computation without having to count all non zeros.
		subBlockRight.setNonZeros(_data.getNumColumns() * nColR);
		MatrixBlock out = new MatrixBlock(_data.getNumRows(), nColR, false);
		LibMatrixMult.matrixMult(_data, subBlockRight, out, InfrastructureAnalyzer.getLocalParallelism());
		return new ColGroupUncompressed(outputCols, out);

	}

	@Override
	public int getNumValues() {
		return _data.getNumRows();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		MatrixBlock replaced = _data.replaceOperations(new MatrixBlock(), pattern, replace);
		return new ColGroupUncompressed(_colIndexes, replaced);
	}
}
