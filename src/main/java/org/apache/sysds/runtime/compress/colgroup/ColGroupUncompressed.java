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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictLibMatrixMult;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.SchemeFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
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
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

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
	private final MatrixBlock _data;

	/**
	 * Do not use this constructor of column group uncompressed, instead use the create constructor.
	 * @param mb The contained data.
	 * @param colIndexes Column indexes for this Columngroup
	 */
	protected ColGroupUncompressed(MatrixBlock mb, IColIndex colIndexes) {
		super(colIndexes);
		_data = mb;
	}

	/**
	 * Do not use this constructor of column group quantization-fused uncompressed, instead use the create constructor.
	 * @param mb The contained data.
	 * @param scaleFactors  For quantization-fused compression, scale factors per row, or a single value for entire matrix
	 * @param colIndexes Column indexes for this Columngroup
	 */
	protected ColGroupUncompressed(MatrixBlock mb, IColIndex colIndexes, double[] scaleFactors) {
		super(colIndexes);
		// Apply scaling and flooring 
		// TODO: Use internal matrix prod 
		for(int r = 0; r < mb.getNumRows(); r++) {
			double scaleFactor = scaleFactors.length == 1 ? scaleFactors[0] : scaleFactors[r];
			for(int c = 0; c < mb.getNumColumns(); c++) {
				double newValue = Math.floor(mb.get(r, c) * scaleFactor);
				mb.set(r, c, newValue);
			}
		}
		_data = mb;
	}	
	/**
	 * Create an Uncompressed Matrix Block, where the columns are offset by col indexes.
	 * 
	 * It is assumed that the size of the colIndexes and number of columns in mb is matching.
	 * 
	 * @param mb         The MB / data to contain in the uncompressed column
	 * @param colIndexes The column indexes for the group
	 * @return An Uncompressed Column group
	 */
	public static AColGroup create(MatrixBlock mb, IColIndex colIndexes) {
		if(mb == null || mb.isEmpty())
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupUncompressed(mb, colIndexes);
	}

	/**
	 * Create ana quantization-fused uncompressed Matrix Block, where the columns are offset by col indexes.
	 * 
	 * It is assumed that the size of the colIndexes and number of columns in mb is matching.
	 * 
	 * @param mb         The MB / data to contain in the uncompressed column
	 * @param colIndexes The column indexes for the group
	 * @param scaleFactors  For quantization-fused compression, scale factors per row, or a single value for entire matrix
	 * @return An Uncompressed Column group
	 */
	public static AColGroup createQuantized(MatrixBlock mb, IColIndex colIndexes, double[] scaleFactors) {
		if(mb == null || mb.isEmpty())
			// TODO: handle quantization-fused compression if deemed necessary,
			// but if the matrix reaches here, it likely doesn't need quantization.
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupUncompressed(mb, colIndexes, scaleFactors);
	}

	/**
	 * Main constructor for a quantization-fused uncompressed ColGroup.
	 * 
	 * @param colIndexes 	Indices (relative to the current block) of the columns that this column group represents.
	 * @param rawBlock   	The uncompressed block; uncompressed data must be present at the time that the constructor is
	 *                   	called
	 * @param transposed 	Says if the input matrix raw block have been transposed.
	 * @param scaleFactors  For quantization-fused compression, scale factors per row, or a single value for entire matrix
	 * @return AColGroup.
	 */
	public static AColGroup createQuantized(IColIndex colIndexes, MatrixBlock rawBlock, boolean transposed, double[] scaleFactors) {

		// special cases
		if(rawBlock.isEmptyBlock(false)) // empty input
			// TODO: handle quantization-fused compression if deemed necessary,
			// but if the matrix reaches here, it likely doesn't need quantization.
			return new ColGroupEmpty(colIndexes);
		else if(!transposed && colIndexes.size() == rawBlock.getNumColumns())
			// full input to uncompressedColumnGroup
			return new ColGroupUncompressed(rawBlock, colIndexes, scaleFactors);

		MatrixBlock mb;
		final int _numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		if(colIndexes.size() == 1) {
			final int col = colIndexes.get(0);
			if(transposed) {
				mb = rawBlock.slice(col, col, 0, rawBlock.getNumColumns() - 1);
				mb = LibMatrixReorg.transposeInPlace(mb, InfrastructureAnalyzer.getLocalParallelism());
			}
			else
				mb = rawBlock.slice(0, rawBlock.getNumRows() - 1, col, col);

			return createQuantized(mb, colIndexes, scaleFactors);
		}

		// Create a matrix with just the requested rows of the original block
		mb = new MatrixBlock(_numRows, colIndexes.size(), rawBlock.isInSparseFormat());

		final int m = _numRows;
		final int n = colIndexes.size();

		if(transposed) {
			if (scaleFactors.length == 1) {
				for(int i = 0; i < m; i++)
					for(int j = 0; j < n; j++)
						mb.appendValue(i, j, Math.floor(rawBlock.get(i, colIndexes.get(j)) * scaleFactors[0]));
			} else {
				for(int i = 0; i < m; i++)
					for(int j = 0; j < n; j++)
						mb.appendValue(i, j, Math.floor(rawBlock.get(i, colIndexes.get(j)) * scaleFactors[j]));
			}
		}
		else {
			if (scaleFactors.length == 1) {
				for(int i = 0; i < m; i++)
					for(int j = 0; j < n; j++)
						mb.appendValue(i, j, Math.floor(rawBlock.get(i, colIndexes.get(j)) * scaleFactors[0]));
			} else {
				for(int i = 0; i < m; i++)
					for(int j = 0; j < n; j++)
						mb.appendValue(i, j, Math.floor(rawBlock.get(i, colIndexes.get(j)) * scaleFactors[i]));
			}
		}

		mb.recomputeNonZeros();
		mb.examSparsity();

		return create(mb, colIndexes);

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
	public static AColGroup create(IColIndex colIndexes, MatrixBlock rawBlock, boolean transposed) {

		// special cases
		if(rawBlock.isEmptyBlock(false)) // empty input
			return new ColGroupEmpty(colIndexes);
		else if(!transposed && colIndexes.size() == rawBlock.getNumColumns())
			// full input to uncompressedColumnGroup
			return new ColGroupUncompressed(rawBlock, colIndexes);

		MatrixBlock mb;
		final int _numRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		if(colIndexes.size() == 1) {
			final int col = colIndexes.get(0);
			if(transposed) {
				mb = rawBlock.slice(col, col, 0, rawBlock.getNumColumns() - 1);
				mb = LibMatrixReorg.transposeInPlace(mb, InfrastructureAnalyzer.getLocalParallelism());
			}
			else
				mb = rawBlock.slice(0, rawBlock.getNumRows() - 1, col, col);

			return create(mb, colIndexes);
		}

		// Create a matrix with just the requested rows of the original block
		mb = new MatrixBlock(_numRows, colIndexes.size(), rawBlock.isInSparseFormat());

		final int m = _numRows;
		final int n = colIndexes.size();

		if(transposed)
			for(int i = 0; i < m; i++)
				for(int j = 0; j < n; j++)
					mb.appendValue(i, j, rawBlock.get(colIndexes.get(j), i));
		else
			for(int i = 0; i < m; i++)
				for(int j = 0; j < n; j++)
					mb.appendValue(i, j, rawBlock.get(i, colIndexes.get(j)));

		mb.recomputeNonZeros();
		mb.examSparsity();

		return create(mb, colIndexes);

	}

	public static AColGroup create(MatrixBlock data) {
		return create(ColIndexFactory.create(data.getNumColumns()), data, false);
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
		return ColGroupSizes.estimateInMemorySizeUncompressed(_data.getNumRows(), _colIndexes.isContiguous(),
			getNumCols(), _data.getSparsity());
	}

	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		// _data is never empty
		if(_data.isInSparseFormat())
			decompressToDenseBlockSparseData(db, rl, ru, offR, offC);
		else if(_colIndexes.size() == db.getDim(1))
			decompressToDenseBlockDenseDataAllColumns(db, rl, ru, offR);
		else
			decompressToDenseBlockDenseData(db, rl, ru, offR, offC);
	}

	private void decompressToDenseBlockDenseData(DenseBlock db, int rl, int ru, int offR, int offC) {
		int offT = rl + offR;
		final int nCol = _colIndexes.size();
		final double[] values = _data.getDenseBlockValues();
		int offS = rl * nCol;
		for(int row = rl; row < ru; row++, offT++, offS += nCol) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes.get(j)] += values[offS + j];
		}
	}

	private void decompressToDenseBlockDenseDataAllColumns(DenseBlock db, int rl, int ru, int offR) {
		if(db.isContiguous() && _data.getDenseBlock().isContiguous())
			decompressToDenseBlockDenseDataAllColumnsContiguous(db, rl, ru, offR);
		else
			decompressToDenseBlockDenseDataAllColumnsGeneric(db, rl, ru, offR);
	}

	private void decompressToDenseBlockDenseDataAllColumnsContiguous(DenseBlock db, int rl, int ru, int offR) {

		final int nCol = _data.getNumColumns();
		final double[] a = _data.getDenseBlockValues();
		final double[] c = db.values(0);
		final int as = rl * nCol;
		final int cs = (rl + offR) * nCol;
		final int sz = ru * nCol - rl * nCol;
		for(int i = 0; i < sz; i += 64) {
			LibMatrixMult.vectAdd(a, c, as + i, cs + i, Math.min(64, sz - i));
		}

	}

	private void decompressToDenseBlockDenseDataAllColumnsGeneric(DenseBlock db, int rl, int ru, int offR) {
		int offT = rl + offR;
		final int nCol = _colIndexes.size();
		DenseBlock tb = _data.getDenseBlock();
		for(int row = rl; row < ru; row++, offT++) {
			final double[] values = tb.values(row);
			final int offS = tb.pos(row);
			final double[] c = db.values(offT);
			final int off = db.pos(offT);
			for(int j = 0; j < nCol; j++) {
				double v = values[offS + j];
				c[off + j] += v;
			}
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
				c[_colIndexes.get(aix[col]) + off] += avals[col];
		}
	}

	@Override
	public void decompressToSparseBlock(SparseBlock ret, int rl, int ru, int offR, int offC) {
		// data is never empty
		if(_data.isInSparseFormat())
			decompressToSparseBlockSparseData(ret, rl, ru, offR, offC);
		else
			decompressToSparseBlockDenseData(ret, rl, ru, offR, offC);
	}

	private void decompressToSparseBlockDenseData(SparseBlock ret, int rl, int ru, int offR, int offC) {
		final int nCol = _colIndexes.size();
		final double[] values = _data.getDenseBlockValues();
		int offS = rl * nCol;
		for(int row = rl, offT = rl + offR; row < ru; row++, offT++, offS += nCol)
			for(int j = 0; j < nCol; j++)
				ret.append(offT, offC + _colIndexes.get(j), values[offS + j]);
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
				ret.append(offT, offC + _colIndexes.get(aix[col]), avals[col]);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _data.get(r, colIdx);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final int nCol = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] retV = result.getDenseBlockValues();
		if(matrix.isInSparseFormat())
			lmmNPSparse(matrix.getSparseBlock(), nCol, retV, nColRet, rl, ru, cl, cu);
		else {
			final DenseBlock db = matrix.getDenseBlock();
			if(db.isContiguous())
				lmmNPDense(db.values(0), nCol, retV, nColRet, rl, ru, cl, cu);
			else
				throw new NotImplementedException(
					"Not implemented support for leftMultByMatrixNoPreAgg non contiguous dense matrix");
		}

	}

	protected void lmmNPSparse(SparseBlock sb, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
		if(cl != 0 || cu != _data.getNumRows())
			throw new NotImplementedException();
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
						retV[offR + _colIndexes.get(aix[i])] += v * aval[i];
				}
			}
		}
		else {
			final double[] dV = _data.getDenseBlockValues();
			final int nColD = _colIndexes.size();
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
						retV[offR + _colIndexes.get(i)] += v * dV[offD + i];
				}
			}
		}
	}

	protected void lmmNPDense(double[] mV, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
		if(_data.isInSparseFormat())
			lmmNPDenseSparse(mV, nCol, retV, nColRet, rl, ru, cl, cu);
		else
			lmmNPDenseDense(mV, nCol, retV, nColRet, rl, ru, cl, cu);
	}

	protected void lmmNPDenseSparse(double[] mV, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
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
					retV[offR + _colIndexes.get(aix[i])] += v * aval[i];
			}
		}
	}

	protected void lmmNPDenseDense(double[] mV, int nCol, double[] retV, int nColRet, int rl, int ru, int cl, int cu) {
		final double[] dV = _data.getDenseBlockValues();
		final int nColD = _colIndexes.size();
		for(int r = rl; r < ru; r++) { // I
			final int off = r * nCol;
			final int offR = r * nColRet;
			for(int c = cl; c < cu; c++) { // K
				lmmNPDenseDenseJ(nColD, c, mV, off, retV, offR, dV);
			}
		}
	}

	private void lmmNPDenseDenseJ(final int nColD, final int c, final double[] mV, final int off, final double[] retV,
		final int offR, final double[] dV) {
		final int offD = c * nColD;
		final double v = mV[off + c];
		for(int i = 0; i < nColD; i++) // J
			retV[offR + _colIndexes.get(i)] += v * dV[offD + i];
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
		LOG.warn("Binary row op left is not supported for Uncompressed Matrix, "
			+ "Implement support for VMr in MatrixBlock Binary Cell operations");
		MatrixBlockDictionary d = MatrixBlockDictionary.create(_data);
		IDictionary dm = d.binOpLeft(op, v, _colIndexes);
		if(dm == null)
			return create(null, _colIndexes);
		else
			return create(((MatrixBlockDictionary) dm).getMatrixBlock(), _colIndexes);
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		MatrixBlock rowVector = Util.extractValues(v, _colIndexes);
		return create(_data.binaryOperations(op, rowVector, null), _colIndexes);
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] result, int nRows, int rl, int ru) {
		final ValueFunction fn = op.aggOp.increOp.fn;
		if(fn instanceof Multiply && op.indexFn instanceof ReduceAll && result[0] == 0)
			return; // product
		else if((fn instanceof Builtin && ((Builtin) fn).getBuiltinCode() == BuiltinCode.MAXINDEX) // index
			|| (fn instanceof CM))
			throw new DMLRuntimeException("Not supported type of Unary Aggregate on colGroup");

		// inefficient since usually uncompressed column groups are used in case of extreme sparsity, it is fine
		// using a slice, since we don't allocate extra just extract the pointers to the sparse rows.

		final MatrixBlock tmpData = (rl == 0 && ru == nRows) ? _data : _data.slice(rl, ru - 1, false);
		MatrixBlock tmp = tmpData.aggregateUnaryOperations(op, new MatrixBlock(), tmpData.getNumRows(),
			new MatrixIndexes(1, 1), true);

		if(tmp.isEmpty()) {
			if(op.aggOp.increOp.fn instanceof Builtin) {
				Builtin b = (Builtin) op.aggOp.increOp.fn;
				if(op.indexFn instanceof ReduceRow)
					for(int i = 0; i < _colIndexes.size(); i++)
						result[_colIndexes.get(i)] = b.execute(result[_colIndexes.get(i)], 0);
				else if(op.indexFn instanceof ReduceAll)
					result[0] = b.execute(result[0], 0);
				else
					for(int row = rl; row < ru; row++)
						result[row] = b.execute(result[row], 0);
			}
			else if(op.aggOp.increOp.fn instanceof Multiply) {
				if(op.indexFn instanceof ReduceRow)
					for(int i = 0; i < _colIndexes.size(); i++)
						result[_colIndexes.get(i)] = 0;
				else if(op.indexFn instanceof ReduceAll)
					result[0] = 0;
				else
					Arrays.fill(result, rl, ru, 0);
			}
			// sum etc.
			return;
		}

		tmp.sparseToDense();
		// The output is always dense in unary aggregates.
		final double[] tmpV = tmp.getDenseBlockValues();

		if(op.aggOp.increOp.fn instanceof Builtin) {
			Builtin b = (Builtin) op.aggOp.increOp.fn;
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes.get(i)] = b.execute(result[_colIndexes.get(i)], tmpV[i]);
			else if(op.indexFn instanceof ReduceAll)
				result[0] = b.execute(result[0], tmpV[0]);
			else
				for(int i = 0, row = rl; row < ru; i++, row++)
					result[row] = b.execute(result[row], tmpV[i]);
		}
		else if(op.aggOp.increOp.fn instanceof Multiply) {
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes.get(i)] = tmpV[i];
			else if(op.indexFn instanceof ReduceAll)
				result[0] *= tmpV[0];
			else
				for(int i = 0, row = rl; row < ru; i++, row++)
					result[row] *= tmpV[i];
		}
		else {
			if(op.indexFn instanceof ReduceRow)
				for(int i = 0; i < tmpV.length; i++)
					result[_colIndexes.get(i)] += tmpV[i];
			else if(op.indexFn instanceof ReduceAll)
				result[0] += tmpV[0];
			else
				for(int i = 0, row = rl; row < ru; i++, row++)
					result[row] += tmpV[i];
		}
	}

	public static ColGroupUncompressed read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		MatrixBlock data = new MatrixBlock();
		data.readFields(in);
		return new ColGroupUncompressed(data, cols);
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
	public double getSum(int nRows) {
		return _data.sum();
	}

	@Override
	public final void tsmm(MatrixBlock ret, int nRows) {

		final int tCol = _colIndexes.size();
		final MatrixBlock tmp = new MatrixBlock(tCol, tCol, true);

		// tsmm but only upper triangle.
		LibMatrixMult.matrixMultTransposeSelf(_data, tmp, true, false);

		if(tmp.isInSparseFormat()) {
			final int numColumns = ret.getNumColumns();
			final double[] result = ret.getDenseBlockValues();
			final SparseBlock sb = tmp.getSparseBlock();
			for(int row = 0; row < tCol; row++) {
				final int offRet = _colIndexes.get(row) * numColumns;
				if(sb.isEmpty(row))
					continue;
				int apos = sb.pos(row);
				int alen = sb.size(row) + apos;
				int[] aix = sb.indexes(row);
				double[] aval = sb.values(row);
				for(int j = apos; j < alen; j++)
					result[offRet + _colIndexes.get(aix[j])] += aval[j];

			}
		}
		else {
			// copy that upper triangle part to ret
			final int numColumns = ret.getNumColumns();
			final double[] result = ret.getDenseBlockValues();
			final double[] tmpV = tmp.getDenseBlockValues();
			for(int row = 0, offTmp = 0; row < tCol; row++, offTmp += tCol) {
				final int offRet = _colIndexes.get(row) * numColumns;
				for(int col = row; col < tCol; col++)
					result[offRet + _colIndexes.get(col)] += tmpV[offTmp + col];
			}
		}
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
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		if(lhs instanceof ColGroupUncompressed)
			leftMultByAColGroupUncompressed((ColGroupUncompressed) lhs, result);
		else if(lhs instanceof APreAgg)
			leftMultByAPreAggColGroup((APreAgg) lhs, result);
		else
			throw new DMLCompressionException("Not supported leftMult colgroup type: " + lhs.getClass().getSimpleName());
	}

	private void leftMultByAPreAggColGroup(APreAgg paCG, MatrixBlock result) {
		final int nCols = paCG.getNumCols();
		final MatrixBlock dictM = paCG._dict.getMBDict(nCols).getMatrixBlock();
		if(dictM == null)
			return;
		if(paCG.getNumCols() != 1) {
			LOG.warn("\nInefficient transpose of uncompressed to fit to"
				+ " t(AColGroup) %*% UncompressedColGroup mult by colGroup uncompressed column"
				+ "\nCurrently solved by t(t(Uncompressed) %*% AColGroup)");
		}
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		final MatrixBlock ucCGT = LibMatrixReorg.transpose(getData(), k);
		final MatrixBlock preAgg = new MatrixBlock(1, paCG.getNumValues(), false);
		final MatrixBlock tmpRes = new MatrixBlock(1, nCols, false);
		preAgg.allocateDenseBlock();
		tmpRes.allocateDenseBlock();
		final int nRowsTransposed = ucCGT.getNumRows();
		final double[] retV = result.getDenseBlockValues();
		final double[] tmpV = tmpRes.getDenseBlockValues();
		final int retCols = result.getNumColumns();

		// Process a row at a time in the transposed block.
		for(int i = 0; i < nRowsTransposed; i++) {
			if(ucCGT.isInSparseFormat() && ucCGT.getSparseBlock().isEmpty(i))
				continue;
			paCG.preAggregate(ucCGT, preAgg.getDenseBlockValues(), i, i + 1);
			preAgg.recomputeNonZeros();
			if(preAgg.isEmpty())
				continue;
			// Fixed ret to enforce that we do not allocate again.
			LibMatrixMult.matrixMult(preAgg, dictM, tmpRes, true);

			final int rowOut = _colIndexes.get(i);
			for(int j = 0; j < nCols; j++) {
				final int colOut = paCG._colIndexes.get(j) * retCols;
				retV[rowOut + colOut] += tmpV[j];
			}
			if(i < nRowsTransposed - 1) {
				preAgg.reset(1, paCG.getPreAggregateSize());
				tmpRes.reset(1, nCols);
			}
		}
	}

	private void leftMultByAColGroupUncompressed(ColGroupUncompressed lhs, MatrixBlock result) {
		final MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.size(), 0);
		final int k = InfrastructureAnalyzer.getLocalParallelism();

		if(lhs._data.getNumColumns() != 1) {
			LOG.warn("Inefficient Left Matrix Multiplication with transpose of left hand side : t(l) %*% r");
		}
		// multiply to temp
		MatrixBlock lhData = lhs._data;
		MatrixBlock transposed = LibMatrixReorg.transpose(lhData, k);
		transposed.setNonZeros(lhData.getNonZeros());
		// do transposed left hand side, matrix multiplication.
		LibMatrixMult.matrixMult(transposed, this._data, tmpRet);

		// add temp to output
		final double[] resV = result.getDenseBlockValues();
		final int nColOut = result.getNumColumns();
		// Guaranteed not empty both sides, therefore safe to not check for empty
		if(tmpRet.isInSparseFormat()) {
			SparseBlock sb = tmpRet.getSparseBlock();
			for(int row = 0; row < lhs._colIndexes.size(); row++) {
				if(sb.isEmpty(row))
					continue;
				final int apos = sb.pos(row);
				final int alen = sb.size(row) + apos;
				final int[] aix = sb.indexes(row);
				final double[] avals = sb.values(row);
				final int offRes = lhs._colIndexes.get(row) * nColOut;
				for(int col = apos; col < alen; col++)
					resV[offRes + _colIndexes.get(aix[col])] += avals[col];
			}
		}
		else {
			final double[] tmpRetV = tmpRet.getDenseBlockValues();
			for(int row = 0; row < lhs._colIndexes.size(); row++) {
				final int offRes = lhs._colIndexes.get(row) * nColOut;
				final int offTmp = _colIndexes.size() * row;
				for(int col = 0; col < _colIndexes.size(); col++)
					resV[offRes + _colIndexes.get(col)] += tmpRetV[offTmp + col];
			}
		}
	}

	@Override
	public void tsmmAColGroup(AColGroup lhs, MatrixBlock result) {
		// this is never empty therefore process:
		if(lhs instanceof ColGroupUncompressed)
			tsmmUncompressedColGroup((ColGroupUncompressed) lhs, result);
		else
			lhs.tsmmAColGroup(this, result);
	}

	private void tsmmUncompressedColGroup(ColGroupUncompressed lhs, MatrixBlock result) {
		final MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.size(), 0);
		final int k = InfrastructureAnalyzer.getLocalParallelism();

		if(lhs._data == this._data)
			LibMatrixMult.matrixMultTransposeSelf(this._data, tmpRet, true, k);
		else {
			LOG.warn("Inefficient Left Matrix Multiplication with transpose of left hand side : t(l) %*% r");
			LibMatrixMult.matrixMult(LibMatrixReorg.transpose(lhs._data, k), this._data, tmpRet);
		}

		final double[] resV = result.getDenseBlockValues();
		final int nCols = result.getNumColumns();
		// guaranteed non empty
		if(tmpRet.isInSparseFormat()) {
			SparseBlock sb = tmpRet.getSparseBlock();
			for(int row = 0; row < lhs._colIndexes.size(); row++) {
				if(sb.isEmpty(row))
					continue;
				final int apos = sb.pos(row);
				final int alen = sb.size(row) + apos;
				final int[] aix = sb.indexes(row);
				final double[] avals = sb.values(row);
				for(int col = apos; col < alen; col++)
					DictLibMatrixMult.addToUpperTriangle(nCols, lhs._colIndexes.get(row), _colIndexes.get(aix[col]), resV,
						avals[col]);
			}
		}
		else {
			double[] tmpRetV = tmpRet.getDenseBlockValues();
			for(int row = 0; row < lhs._colIndexes.size(); row++) {
				final int offTmp = lhs._colIndexes.size() * row;
				final int oid = lhs._colIndexes.get(row);
				for(int col = 0; col < _colIndexes.size(); col++)
					DictLibMatrixMult.addToUpperTriangle(nCols, oid, _colIndexes.get(col), resV, tmpRetV[offTmp + col]);
			}
		}
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		return sliceMultiColumns(idx, idx + 1, ColIndexFactory.create(1));
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		MatrixBlock newData = _data.slice(0, _data.getNumRows() - 1, idStart, idEnd - 1, true);
		return create(newData, outputCols);
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		final int nColR = right.getNumColumns();
		final IColIndex outputCols = allCols != null ? allCols : ColIndexFactory.create(nColR);

		if(right.isEmpty())
			return null;

		MatrixBlock subBlockRight;

		if(right.isInSparseFormat()) {
			subBlockRight = new MatrixBlock(_data.getNumColumns(), nColR, true);
			subBlockRight.allocateSparseRowsBlock();
			final SparseBlock sbR = right.getSparseBlock();
			final SparseBlock subR = subBlockRight.getSparseBlock();
			long nnz = 0;
			for(int i = 0; i < _colIndexes.size(); i++) {
				if(sbR.isEmpty(_colIndexes.get(i)))
					continue;
				subR.set(i, sbR.get(_colIndexes.get(i)), false);
				nnz += sbR.get(_colIndexes.get(i)).size();
			}
			subBlockRight.setNonZeros(nnz);
		}
		else {
			subBlockRight = new MatrixBlock(_data.getNumColumns(), nColR, false);
			subBlockRight.allocateDenseBlock();
			final double[] sbr = subBlockRight.getDenseBlockValues();
			final double[] rightV = right.getDenseBlockValues();
			for(int i = 0; i < _colIndexes.size(); i++) {
				final int offSubBlock = i * nColR;
				final int offRight = _colIndexes.get(i) * nColR;
				System.arraycopy(rightV, offRight, sbr, offSubBlock, nColR);
			}
			subBlockRight.setNonZeros(_data.getNumColumns() * nColR);
		}
		MatrixBlock out = new MatrixBlock(_data.getNumRows(), nColR, false);
		LibMatrixMult.matrixMult(_data, subBlockRight, out, k);
		return create(out, outputCols);

	}

	@Override
	public int getNumValues() {
		return _data.getNumRows();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		if(Util.eq(pattern, Double.NaN) && !_data.containsValue(pattern)) {
			return this; // return this.
		}
		MatrixBlock replaced = _data.replaceOperations(new MatrixBlock(), pattern, replace);
		return create(replaced, _colIndexes);
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		final MatrixBlock colSum = _data.colSum();
		if(colSum.isInSparseFormat()) {
			SparseBlock sb = colSum.getSparseBlock();
			double[] rv = sb.values(0);
			int[] idx = sb.indexes(0);
			for(int i = 0; i < idx.length; i++)
				c[_colIndexes.get(idx[i])] += rv[i];
		}
		else {
			double[] dv = colSum.getDenseBlockValues();
			for(int i = 0; i < _colIndexes.size(); i++)
				c[_colIndexes.get(i)] += dv[i];
		}
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		return _data.cmOperations(op);
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		MatrixBlock nd = LibMatrixReorg.rexpand(_data, new MatrixBlock(), max, false, cast, ignore, 1);
		return create(nd, ColIndexFactory.create(max));
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(nRows, nRows, nCols, nVals, _data.getSparsity());
	}

	@Override
	public boolean isEmpty() {
		return _data.isEmpty();
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		final MatrixBlock mb = _data.slice(rl, ru - 1);
		if(mb.isEmpty())
			return null;
		return new ColGroupUncompressed(mb, _colIndexes);
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupUncompressed && g.getColIndices().equals(_colIndexes)) {
			final ColGroupUncompressed gDDC = (ColGroupUncompressed) g;
			final MatrixBlock nd = _data.append(gDDC._data, false);
			return create(nd, _colIndexes);
		}
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		final MatrixBlock ret = new MatrixBlock(rlen, _colIndexes.size(), _data.isInSparseFormat());
		ret.allocateBlock();
		final SparseBlock sb = ret.getSparseBlock();
		final DenseBlock db = ret.getDenseBlock();
		final IColIndex target = ColIndexFactory.create(_colIndexes.size());
		for(int i = 0; i < g.length; i++) {
			final int start = i * blen;
			final int end = Math.min(i * blen + blen, rlen);
			final AColGroup gs = g[i];
			if(_data.isInSparseFormat())
				gs.copyAndSet(target).decompressToSparseBlock(sb, 0, end - start, start, 0);
			else
				gs.copyAndSet(target).decompressToDenseBlock(db, 0, end - start, start, 0);
		}
		ret.recomputeNonZeros();
		return new ColGroupUncompressed(ret, _colIndexes);
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return SchemeFactory.create(_colIndexes, CompressionType.UNCOMPRESSED);
	}

	@Override
	public AColGroup recompress() {

		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		final CompressionSettings cs = new CompressionSettingsBuilder().create();
		final EstimationFactors f = new EstimationFactors(_data.getNumRows(), _data.getNumRows(), _data.getSparsity());
		es.add(new CompressedSizeInfoColGroup( //
			ColIndexFactory.create(_data.getNumColumns()), f, 312152, CompressionType.DDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);
		final List<AColGroup> comp = ColGroupFactory.compressColGroups(_data, csi, cs);

		return comp.get(0).copyAndSet(_colIndexes);
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		// final IEncode map = EncodingFactory.createFromMatrixBlock(_data, false,
		// ColIndexFactory.create(_data.getNumColumns()));
		final int _numRows = _data.getNumRows();
		final CompressionSettings _cs = new CompressionSettingsBuilder().create();// default settings
		final EstimationFactors em = new EstimationFactors(_numRows, _numRows, 1, null, _numRows, _numRows, _numRows,
			false, false, (double) _numRows / _data.getNonZeros(), (double) _numRows / _data.getNonZeros());
		// map.extractFacts(_numRows, _data.getSparsity(), _data.getSparsity(), _cs);
		return new CompressedSizeInfoColGroup(_colIndexes, em, _cs.validCompressions, null);
	}

	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		return new ColGroupUncompressed(_data, colIndexes);
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		MatrixBlock ret = new MatrixBlock(_data.getNumRows(), _data.getNumColumns(), _data.getNonZeros());
		// TODO add sparse optmization
		for(int r = 0; r < _data.getNumRows(); r++)
			for(int c = 0; c < _data.getNumColumns(); c++)
				ret.set(r, c, _data.get(r, reordering[c]));
		return create(newColIndex, ret, false);
	}

	@Override
	public double getSparsity() {
		return _data.getSparsity();
	}

	@Override
	public AColGroup morph(CompressionType ct, int nRow) {
		if(ct == getCompType())
			return this;

		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		final CompressionSettings cs = new CompressionSettingsBuilder().create();
		final EstimationFactors f = new EstimationFactors(_data.getNumRows(), _data.getNumRows(), _data.getSparsity());
		es.add(new CompressedSizeInfoColGroup(ColIndexFactory.create(_data.getNumColumns()), f, 312152, ct));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);
		final List<AColGroup> comp = ColGroupFactory.compressColGroups(_data, csi, cs);

		return comp.get(0).copyAndSet(_colIndexes);
	}


	@Override
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		if(_data.isInSparseFormat())
			sparseSelectionSparseColumnGroup(selection, ret, rl, ru);
		else
			sparseSelectionDenseColumnGroup(selection, ret, rl, ru);
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		if(_data.isInSparseFormat())
			denseSelectionSparseColumnGroup(selection, ret, rl, ru);
		else
			denseSelectionDenseColumnGroup(selection, ret, rl, ru);
	}


	private void sparseSelectionSparseColumnGroup(MatrixBlock selection, MatrixBlock ret, int rl, int ru) {

		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		final SparseBlock tb = _data.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			if(tb.isEmpty(rowCompressed))
				continue;
			final int tPos = tb.pos(rowCompressed);
			final int tEnd = tb.size(rowCompressed) + tPos;
			final int[] tIx = tb.indexes(rowCompressed);
			final double[] tVal = tb.values(rowCompressed);
			for(int j = tPos; j < tEnd; j++)
				retB.append(r, _colIndexes.get(tIx[j]), tVal[j]);
		}

	}

	private void sparseSelectionDenseColumnGroup(MatrixBlock selection, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		final DenseBlock tb = _data.getDenseBlock();
		final int nCol = _colIndexes.size();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];

			double[] tVal = tb.values(rowCompressed);
			int tPos = tb.pos(rowCompressed);
			for(int j = 0; j < nCol; j++)
				retB.append(r, _colIndexes.get(j), tVal[tPos + j]);
		}
	}

	private void denseSelectionSparseColumnGroup(MatrixBlock selection, MatrixBlock ret, int rl, int ru) {

		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		final SparseBlock tb = _data.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			if(tb.isEmpty(rowCompressed))
				continue;
			final int tPos = tb.pos(rowCompressed);
			final int tEnd = tb.size(rowCompressed) + tPos;
			final int[] tIx = tb.indexes(rowCompressed);
			final double[] tVal = tb.values(rowCompressed);

			final double[] rVal = retB.values(r);
			final int pos = retB.pos(r);
			for(int j = tPos; j < tEnd; j++)
				rVal[pos + _colIndexes.get(tIx[j])] += tVal[j];
		}

	}

	private void denseSelectionDenseColumnGroup(MatrixBlock selection, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		final DenseBlock tb = _data.getDenseBlock();
		final int nCol = _colIndexes.size();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];

			double[] tVal = tb.values(rowCompressed);
			int tPos = tb.pos(rowCompressed);

			final double[] rVal = retB.values(r);
			final int pos = retB.pos(r);

			for(int j = 0; j < nCol; j++)
				rVal[pos + _colIndexes.get(j)] += tVal[tPos + j];
		}
	}

	@Override
	public AColGroup reduceCols() {
		MatrixBlock mb = _data.rowSum();
		if(mb.isEmpty())
			return null;
		else
			return new ColGroupUncompressed(mb, ColIndexFactory.createI(0));
	}
	
	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		if(_data.isInSparseFormat())
			decompressToDenseBlockTransposedSparse(db, rl, ru);
		else
			decompressToDenseBlockTransposedDense(db, rl, ru);
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		if(_data.isInSparseFormat())
			decompressToSparseBlockTransposedSparse(sb);
		else
			decompressToSparseBlockTransposedDense(sb);
	}

	private void decompressToSparseBlockTransposedSparse(SparseBlock sb) {
		MatrixBlock transposedDict = LibMatrixReorg.transpose(_data);
		if(transposedDict.isInSparseFormat()) {
			SparseBlock sbThis = transposedDict.getSparseBlock();
			for(int i = 0; i < _colIndexes.size(); i++) {
				sb.set(_colIndexes.get(i), sbThis.get(i), false);
			}
		}
		else {
			throw new NotImplementedException();
		}
	}

	private void decompressToSparseBlockTransposedDense(SparseBlockMCSR sb) {
		DenseBlock dbThis = _data.getDenseBlock();
		if(!dbThis.isContiguous())
			throw new NotImplementedException("Not Implemented transposed decompress on non contiguous matrix");
		final int colsThis = _colIndexes.size();
		final int rowsThis = _data.getNumRows();
		double[] valsThis = dbThis.valuesAt(0);
		if(colsThis == 1) {
			sb.allocate(_colIndexes.get(0), (int) _data.getNonZeros());
		}
		else {
			for(int c = 0; c < colsThis; c++) {
				sb.allocate(_colIndexes.get(0), Math.max(2, (int) (_data.getNonZeros() / colsThis)));
			}
		}

		for(int c = 0; c < colsThis; c++) {
			final int rowOut = _colIndexes.get(c);
			SparseRow sbr = sb.get(rowOut);
			if(sbr == null)
				sbr = new SparseRowVector(4);
			for(int r = 0; r < rowsThis; r++)
				sbr = sbr.append(rowOut, valsThis[colsThis * r + c]);
			sb.set(c, sbr, false);
		}
	}

	private void decompressToDenseBlockTransposedSparse(DenseBlock db, int rl, int ru) {
		throw new NotImplementedException();
	}

	private void decompressToDenseBlockTransposedDense(DenseBlock db, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public boolean sameIndexStructure(AColGroup that) {
		return that instanceof ColGroupUncompressed &&
			((ColGroupUncompressed) that)._data.getNumRows() == _data.getNumRows();
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		ColGroupUncompressed rightUC = ((ColGroupUncompressed) right);
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		MatrixBlock combined = _data.append(rightUC._data, null, true);
		return new ColGroupUncompressed(combined, combinedColIndex);
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		MatrixBlock[] cbindOther = new MatrixBlock[right.size()];
		for(int i = 0; i < right.size(); i++) {
			cbindOther[i] = ((ColGroupUncompressed) right.get(i))._data;
		}
		final MatrixBlock combined = _data.append(cbindOther, null, true);
		return new ColGroupUncompressed(combined, combinedColIndex);
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		final int s = _colIndexes.size();
		final int[] newColumns = new int[s * multiplier];
		for(int i = 0; i < multiplier; i++)
			for(int j = 0; j < s; j++)
				newColumns[i * s + j] = _colIndexes.get(j) + nColOrg * i;
		MatrixBlock newData = _data.reshape(nRow/ multiplier, s * multiplier, true);
		return new AColGroup[]{create(newData,ColIndexFactory.create(newColumns))};
		// throw new NotImplementedException("Unimplemented method 'splitReshape'");
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

		sb.append("\n");
		if(_data.getNumRows() < 1000)
			sb.append(_data.toString());
		else
			sb.append(" don't print uncompressed matrix because it is to big.");

		return sb.toString();
	}
}
