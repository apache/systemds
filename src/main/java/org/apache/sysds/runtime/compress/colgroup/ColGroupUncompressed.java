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
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictLibMatrixMult;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Column group type for columns that are stored as dense arrays of doubles. Uses a MatrixBlock internally to store the
 * column contents.
 */
public class ColGroupUncompressed extends AColGroup {
	private static final long serialVersionUID = -8254271148043292199L;
	/**
	 * We store the contents of the columns as a MatrixBlock to take advantage of high-performance routines available for
	 * this data structure.
	 */
	private MatrixBlock _data;

	/** Constructor for serialization */
	protected ColGroupUncompressed() {
		super();
	}

	private ColGroupUncompressed(MatrixBlock mb, int[] colIndexes) {
		super(colIndexes);
		_data = mb;
	}

	protected static AColGroup create(MatrixBlock mb, int[] colIndexes) {
		if(mb.isEmpty())
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupUncompressed(mb, colIndexes);
	}

	/**
	 * Main constructor for Uncompressed ColGroup.
	 * 
	 * @param colIndexes Indices (relative to the current block) of the columns that this column group represents.
	 * @param rawBlock   The uncompressed block; uncompressed data must be present at the time that the constructor is
	 *                   called
	 * @param transposed Says if the input matrix raw block have been transposed.
	 * @return AColGroup.
	 */
	public static AColGroup create(int[] colIndexes, MatrixBlock rawBlock, boolean transposed) {

		// special cases
		if(rawBlock.isEmptyBlock(false)) // empty input
			return new ColGroupEmpty(colIndexes);
		else if(!transposed && colIndexes.length == rawBlock.getNumColumns())
			// full input to uncompressedColumnGroup
			return new ColGroupUncompressed(rawBlock, colIndexes);

		MatrixBlock mb;
		final int _numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		if(colIndexes.length == 1) {
			final int col = colIndexes[0];
			if(transposed) {
				mb = rawBlock.slice(col, col, 0, rawBlock.getNumColumns() - 1);
				mb = LibMatrixReorg.transposeInPlace(mb, InfrastructureAnalyzer.getLocalParallelism());
			}
			else
				mb = rawBlock.slice(0, rawBlock.getNumRows() - 1, col, col);

			return create(mb, colIndexes);
		}

		// Create a matrix with just the requested rows of the original block
		mb = new MatrixBlock(_numRows, colIndexes.length, rawBlock.isInSparseFormat());

		final int m = _numRows;
		final int n = colIndexes.length;

		if(transposed)
			for(int i = 0; i < m; i++)
				for(int j = 0; j < n; j++)
					mb.appendValue(i, j, rawBlock.quickGetValue(colIndexes[j], i));
		else
			for(int i = 0; i < m; i++)
				for(int j = 0; j < n; j++)
					mb.appendValue(i, j, rawBlock.quickGetValue(i, colIndexes[j]));

		mb.recomputeNonZeros();
		mb.examSparsity();

		return create(mb, colIndexes);

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
		_data.recomputeNonZeros();
	}

	/**
	 * Constructor for allocating a single uncompressed column group.
	 * 
	 * @param data matrix block
	 */
	public ColGroupUncompressed(MatrixBlock data) {
		super(Util.genColsIndices(data.getNumColumns()));
		_data = data;
		_data.recomputeNonZeros();
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
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		if(_data.isEmpty())
			return;
		else if(_data.isInSparseFormat())
			decompressToDenseBlockSparseData(db, rl, ru, offR, offC);
		else
			decompressToDenseBlockDenseData(db, rl, ru, offR, offC);
	}

	private void decompressToDenseBlockDenseData(DenseBlock db, int rl, int ru, int offR, int offC) {
		int offT = rl + offR;
		final int nCol = _colIndexes.length;
		final double[] values = _data.getDenseBlockValues();
		int offS = rl * nCol;
		for(int row = rl; row < ru; row++, offT++, offS += nCol) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[offS + j];
		}
	}

	private void decompressToDenseBlockSparseData(DenseBlock db, int rl, int ru, int offR, int offC) {

		final SparseBlock sb = _data.getSparseBlock();
		for(int row = rl, offT = rl + offR; row < ru; row++, offT++) {
			if(sb.isEmpty(row))
				continue;
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int apos = sb.pos(row);
			final int alen = sb.size(row) + apos;
			final int[] aix = sb.indexes(row);
			final double[] avals = sb.values(row);
			for(int col = apos; col < alen; col++)
				c[_colIndexes[aix[col]] + off] += avals[col];
		}
	}

	@Override
	public void decompressToSparseBlock(SparseBlock ret, int rl, int ru, int offR, int offC) {
		if(_data.isEmpty())
			return;
		else if(_data.isInSparseFormat())
			decompressToSparseBlockSparseData(ret, rl, ru, offR, offC);
		else
			decompressToSparseBlockDenseData(ret, rl, ru, offR, offC);
	}

	private void decompressToSparseBlockDenseData(SparseBlock ret, int rl, int ru, int offR, int offC) {
		final int nCol = _colIndexes.length;
		final double[] values = _data.getDenseBlockValues();
		int offS = rl * nCol;
		for(int row = rl, offT = rl + offR; row < ru; row++, offT++, offS += nCol)
			for(int j = 0; j < nCol; j++)
				ret.append(offT, offC + _colIndexes[j], values[offS + j]);
	}

	private void decompressToSparseBlockSparseData(SparseBlock ret, int rl, int ru, int offR, int offC) {
		int offT = rl + offR;
		final SparseBlock sb = _data.getSparseBlock();
		for(int row = rl; row < ru; row++, offT++) {
			if(sb.isEmpty(row))
				continue;
			final int apos = sb.pos(row);
			final int alen = sb.size(row) + apos;
			final int[] aix = sb.indexes(row);
			final double[] avals = sb.values(row);
			for(int col = apos; col < alen; col++)
				ret.append(offT, offC + _colIndexes[aix[col]], avals[col]);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _data.quickGetValue(r, colIdx);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final int nCol = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] retV = result.getDenseBlockValues();
		if(matrix.isInSparseFormat())
			lmmNPSparse(matrix.getSparseBlock(), nCol, retV, nColRet, rl, ru, cl, cu);
		else
			lmmNPDense(matrix.getDenseBlockValues(), nCol, retV, nColRet, rl, ru, cl, cu);

	}

	protected void lmmNPSparse(SparseBlock sb, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
		if(_data.isInSparseFormat()) {
			final SparseBlock dsb = _data.getSparseBlock();
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final int aposL = sb.pos(r);
				final int alenL = sb.size(r) + aposL;
				final int[] aixL = sb.indexes(r);
				final double[] avalL = sb.values(r);
				final int offR = r * nColRet;
				for(int j = aposL; j < alenL; j++) {
					final int c = aixL[j];
					if(dsb.isEmpty(c))
						continue;
					final double v = avalL[j];
					final int apos = dsb.pos(c);
					final int alen = dsb.size(c) + apos;
					final int[] aix = dsb.indexes(c);
					final double[] aval = dsb.values(c);
					for(int i = apos; i < alen; i++)
						retV[offR + _colIndexes[aix[i]]] += v * aval[i];
				}
			}
		}
		else {
			final double[] dV = _data.getDenseBlockValues();
			final int nColD = _colIndexes.length;
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final int aposL = sb.pos(r);
				final int alenL = sb.size(r) + aposL;
				final int[] aixL = sb.indexes(r);
				final double[] avalL = sb.values(r);
				final int offR = r * nColRet;
				for(int j = aposL; j < alenL; j++) {
					final int c = aixL[j];
					final double v = avalL[j];
					final int offD = c * nColD;
					for(int i = 0; i < nColD; i++)
						retV[offR + _colIndexes[i]] += v * dV[offD + i];
				}
			}
		}
	}

	protected void lmmNPDense(double[] mV, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int r = rl; r < ru; r++) {
				final int off = r * nCol;
				final int offR = r * nColRet;
				for(int c = cl; c < cu; c++) {
					if(sb.isEmpty(c))
						continue;
					final int apos = sb.pos(c);
					final int alen = sb.size(c) + apos;
					final int[] aix = sb.indexes(c);
					final double[] aval = sb.values(c);
					final double v = mV[off + c];
					for(int i = apos; i < alen; i++)
						retV[offR + _colIndexes[aix[i]]] += v * aval[i];

				}
			}
		}
		else {
			final double[] dV = _data.getDenseBlockValues();
			final int nColD = _colIndexes.length;
			for(int r = rl; r < ru; r++) {
				final int off = r * nCol;
				final int offR = r * nColRet;
				for(int c = cl; c < cu; c++) {
					final int offD = c * nColD;
					final double v = mV[off + c];
					for(int i = 0; i < nColD; i++)
						retV[offR + _colIndexes[i]] += v * dV[offD + i];
				}
			}
		}
	}

//	 @Override
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
		MatrixBlock retContent = _data.scalarOperations(op, new MatrixBlock());
		return create(retContent, getColIndices());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		MatrixBlock retContent = _data.unaryOperations(op, new MatrixBlock());
		return create(retContent, getColIndices());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException("Binary row op left is not supported for Uncompressed Matrix, "
			+ "Implement support for VMr in MatrixBlock Binary Cell operations");
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		MatrixBlock rowVector = Util.extractValues(v, _colIndexes);
		return create(_data.binaryOperations(op, rowVector, null), getColIndices());
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] result, int nRows, int rl, int ru) {
		if(op.aggOp.increOp.fn instanceof Multiply && op.indexFn instanceof ReduceAll && result[0] == 0) {
			return; // product
		}

		// LOG.warn("Inefficient Unary Aggregate because of Uncompressed ColumnGroup");
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
		else if(op.aggOp.increOp.fn instanceof Multiply) {
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes[i]] *= tmpV[i];
			else if(op.indexFn instanceof ReduceAll)
				result[0] *= tmpV[0];
			else
				for(int i = 0, row = rl; i < tmpV.length; i++, row++)
					result[row] *= tmpV[i];
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
		super.readFields(in);
		_data = new MatrixBlock();
		_data.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		return super.getExactSizeOnDisk() + _data.getExactSizeOnDisk();
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
	public final void tsmm(MatrixBlock ret, int nRows) {
		if(_data.isEmpty())
			return; // early abort
		final int tCol = _colIndexes.length;
		final MatrixBlock tmp = new MatrixBlock(tCol, tCol, true);

		// tsmm but only upper triangle.
		LibMatrixMult.matrixMultTransposeSelf(_data, tmp, true, false);

		// copy that upper triangle part to ret
		final int numColumns = ret.getNumColumns();
		final double[] result = ret.getDenseBlockValues();
		final double[] tmpV = tmp.getDenseBlockValues();
		for(int row = 0, offTmp = 0; row < tCol; row++, offTmp += tCol) {
			final int offRet = _colIndexes[row] * numColumns;
			for(int col = row; col < tCol; col++)
				result[offRet + _colIndexes[col]] += tmpV[offTmp + col];
		}
	}

	@Override
	public AColGroup copy() {
		return create(_data, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		return _data.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		return _data.getNonZeros();
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		if(lhs instanceof ColGroupEmpty || getData().isEmpty())
			return;
		else if(lhs instanceof ColGroupUncompressed) {
			ColGroupUncompressed lhsUC = (ColGroupUncompressed) lhs;
			MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.length, 0);

			if(lhsUC._data == this._data) {

				LibMatrixMult.matrixMultTransposeSelf(this._data, tmpRet, true,
					InfrastructureAnalyzer.getLocalParallelism());
			}
			else {
				LOG.warn("Inefficient Left Matrix Multiplication with transpose of left hand side : t(l) %*% r");
				MatrixBlock lhData = lhsUC._data;
				MatrixBlock transposed = LibMatrixReorg.transpose(lhData, InfrastructureAnalyzer.getLocalParallelism());
				transposed.setNonZeros(lhData.getNonZeros());
				// do transposed left hand side, matrix multiplication.
				LibMatrixMult.matrixMult(transposed, this._data, tmpRet);
			}

			ColGroupUtils.copyValuesColGroupMatrixBlocks(lhs, this, tmpRet, result);
		}
		else if(lhs instanceof APreAgg) {
			// throw new NotImplementedException();
			LOG.warn("\nInefficient transpose of uncompressed to fit to"
				+ " t(AColGroup) %*% UncompressedColGroup mult by colGroup uncompressed column"
				+ "\nCurrently solved by t(t(Uncompressed) %*% AColGroup)");

			final MatrixBlock ucCGT = LibMatrixReorg.transpose(getData(), InfrastructureAnalyzer.getLocalParallelism());

			final APreAgg paCG = (APreAgg) lhs;
			final MatrixBlock preAgg = new MatrixBlock(1, lhs.getNumValues(), false);
			final MatrixBlock tmpRes = new MatrixBlock(1, this.getNumCols(), false);
			final MatrixBlock dictM = paCG._dict.getMBDict(paCG.getNumCols()).getMatrixBlock();
			preAgg.allocateDenseBlock();
			tmpRes.allocateDenseBlock();
			final int nRows = ucCGT.getNumRows();
			final int nCols = lhs.getNumCols();
			final double[] retV = result.getDenseBlockValues();
			final double[] tmpV = tmpRes.getDenseBlockValues();
			final int retCols = result.getNumColumns();
			for(int i = 0; i < nRows; i++) {
				if(ucCGT.isInSparseFormat() && ucCGT.getSparseBlock().isEmpty(i))
					continue;
				paCG.preAggregate(ucCGT, preAgg.getDenseBlockValues(), i, i + 1);
				preAgg.recomputeNonZeros();
				LibMatrixMult.matrixMult(preAgg, dictM, tmpRes, true);

				final int rowOut = _colIndexes[i];
				for(int j = 0; j < nCols; j++) {
					final int colOut = lhs._colIndexes[j] * retCols;
					retV[rowOut + colOut] += tmpV[j];
				}
				if(i < nRows - 1) {
					preAgg.reset(1, lhs.getNumValues());
					tmpRes.reset(1, this.getNumCols());
				}
			}
		}
		else {
			throw new NotImplementedException();
		}
	}

	@Override
	public void tsmmAColGroup(AColGroup lhs, MatrixBlock result) {
		if(this._data.isEmpty())
			return; // early abort
		if(lhs instanceof ColGroupUncompressed) {
			ColGroupUncompressed lhsUC = (ColGroupUncompressed) lhs;
			if(lhsUC._data.isEmpty())
				return; // early abort

			MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.length, 0);
			if(lhsUC._data == this._data) {
				LibMatrixMult.matrixMultTransposeSelf(this._data, tmpRet, true,
					InfrastructureAnalyzer.getLocalParallelism());
			}
			else {
				LOG.warn("Inefficient Left Matrix Multiplication with transpose of left hand side : t(l) %*% r");
				MatrixBlock lhData = lhsUC._data;
				MatrixBlock transposed = LibMatrixReorg.transpose(lhData, InfrastructureAnalyzer.getLocalParallelism());
				transposed.setNonZeros(lhData.getNonZeros());
				// do transposed left hand side, matrix multiplication.
				LibMatrixMult.matrixMult(transposed, this._data, tmpRet);
			}
			final double[] resV = result.getDenseBlockValues();
			final int nCols = result.getNumColumns();
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
					for(int col = apos; col < alen; col++)
						DictLibMatrixMult.addToUpperTriangle(nCols, lhs._colIndexes[row], _colIndexes[aix[col]], resV,
							avals[col]);
				}
			}
			else {
				double[] tmpRetV = tmpRet.getDenseBlockValues();
				for(int row = 0; row < lhs._colIndexes.length; row++) {
					final int offTmp = lhs._colIndexes.length * row;
					for(int col = 0; col < _colIndexes.length; col++)
						DictLibMatrixMult.addToUpperTriangle(nCols, lhs._colIndexes[row], _colIndexes[col], resV,
							tmpRetV[offTmp + col]);
				}
			}
		}
		else
			lhs.tsmmAColGroup(this, result);
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		return sliceMultiColumns(idx, idx + 1, new int[] {0});
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		MatrixBlock newData = _data.slice(0, _data.getNumRows() - 1, idStart, idEnd - 1, true);
		return create(newData, outputCols);
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		final int nColR = right.getNumColumns();
		final int[] outputCols = Util.genColsIndices(nColR);

		if(_data.isEmpty() || right.isEmpty())
			return new ColGroupEmpty(outputCols);

		MatrixBlock subBlockRight;

		if(right.isInSparseFormat()) {
			subBlockRight = new MatrixBlock(_data.getNumColumns(), nColR, true);
			subBlockRight.allocateSparseRowsBlock();
			final SparseBlock sbR = right.getSparseBlock();
			final SparseBlock subR = subBlockRight.getSparseBlock();
			long nnz = 0;
			for(int i = 0; i < _colIndexes.length; i++) {
				if(sbR.isEmpty(_colIndexes[i]))
					continue;
				subR.set(i, sbR.get(_colIndexes[i]), false);
				nnz += sbR.get(_colIndexes[i]).size();
			}
			subBlockRight.setNonZeros(nnz);
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
			subBlockRight.setNonZeros(_data.getNumColumns() * nColR);
		}
		MatrixBlock out = new MatrixBlock(_data.getNumRows(), nColR, false);
		LibMatrixMult.matrixMult(_data, subBlockRight, out, InfrastructureAnalyzer.getLocalParallelism());
		return create(out, outputCols);

	}

	@Override
	public int getNumValues() {
		return _data.getNumRows();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		MatrixBlock replaced = _data.replaceOperations(new MatrixBlock(), pattern, replace);
		return create(replaced, _colIndexes);
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		final MatrixBlock colSum = _data.colSum();
		if(colSum.isEmpty())
			return;
		else if(colSum.isInSparseFormat()) {
			SparseBlock sb = colSum.getSparseBlock();
			double[] rv = sb.values(0);
			int[] idx = sb.indexes(0);
			for(int i = 0; i < idx.length; i++)
				c[_colIndexes[idx[i]]] += rv[i];
		}
		else {
			double[] dv = colSum.getDenseBlockValues();
			for(int i = 0; i < _colIndexes.length; i++)
				c[_colIndexes[i]] += dv[i];
		}
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		return _data.cmOperations(op);
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		MatrixBlock nd = LibMatrixReorg.rexpand(_data, new MatrixBlock(), max, false, cast, ignore, 1);
		return create(nd, Util.genColsIndices(max));
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(nRows, nRows, nCols, nVals, _data.getSparsity());
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\n");
		sb.append(" numCols : " + _data.getNumColumns());
		sb.append(" numRows : " + _data.getNumRows());
		sb.append(" nonZeros: " + _data.getNonZeros());
		sb.append(" Sparse : " + _data.isInSparseFormat());
		if(_data.isEmpty()) {
			sb.append(" empty");
			return sb.toString();
		}

		sb.append("\n");
		if(!_data.isInSparseFormat() && _data.getNumRows() < 1000)
			sb.append(Arrays.toString(_data.getDenseBlockValues()));
		else if(_data.getNumRows() < 100)
			sb.append(_data.toString());
		else
			sb.append(" don't print uncompressed matrix because it is to big.");

		return sb.toString();
	}
}
