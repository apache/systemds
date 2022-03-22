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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 */
public class ColGroupDDC extends APreAgg {
	private static final long serialVersionUID = -5769772089913918987L;

	protected AMapToData _data;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows number of rows
	 */
	protected ColGroupDDC(int numRows) {
		super(numRows);
	}

	private ColGroupDDC(int[] colIndexes, int numRows, ADictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, numRows, dict, cachedCounts);
		if(data.getUnique() != dict.getNumberOfValues(colIndexes.length))
			throw new DMLCompressionException("Invalid construction of DDC group " + data.getUnique() + " vs. "
				+ dict.getNumberOfValues(colIndexes.length));
		_zeros = false;
		_data = data;

	}

	protected static AColGroup create(int[] colIndexes, int numRows, ADictionary dict, AMapToData data,
		int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupDDC(colIndexes, numRows, dict, data, cachedCounts);
	}

	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		if(db.isContiguous() && _colIndexes.length == 1) {
			if(db.getDim(1) == 1)
				decompressToDenseBlockDenseDictSingleColOutContiguous(db, rl, ru, offR, offC, values);
			else
				decompressToDenseBlockDenseDictSingleColContiguous(db, rl, ru, offR, offC, values);
		}
		else if(db.isContiguous() && _colIndexes.length == db.getDim(1) && offC == 0)
			decompressToDenseBlockDenseDictAllColumnsContiguous(db, rl, ru, offR, values);
		else
			decompressToDenseBlockDenseDictGeneric(db, rl, ru, offR, offC, values);

	}

	private void decompressToDenseBlockDenseDictSingleColContiguous(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final double[] c = db.values(0);
		final int nCols = db.getDim(1);
		final int colOff = _colIndexes[0] + offC;
		for(int i = rl, offT = (rl + offR) * nCols + colOff; i < ru; i++, offT += nCols)
			c[offT] += values[_data.getIndex(i)];

	}

	private void decompressToDenseBlockDenseDictSingleColOutContiguous(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final double[] c = db.values(0);
		for(int i = rl, offT = rl + offR + _colIndexes[0] + offC; i < ru; i++, offT++)
			c[offT] += values[_data.getIndex(i)];
	}

	private void decompressToDenseBlockDenseDictAllColumnsContiguous(DenseBlock db, int rl, int ru, int offR,
		double[] values) {
		final double[] c = db.values(0);
		final int nCol = _colIndexes.length;
		for(int r = rl; r < ru; r++) {
			final int start = _data.getIndex(r) * nCol;
			final int end = start + nCol;
			final int offStart = (offR + r) * nCol;
			for(int vOff = start, off = offStart; vOff < end; vOff++, off++)
				c[off] += values[vOff];
		}
	}

	private void decompressToDenseBlockDenseDictGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		// generic
		final int nCol = _colIndexes.length;
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[rowIndex + j];
		}
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.length;
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes[j] + offC, values[rowIndex + j]);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r) * _colIndexes.length + colIdx);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] += preAgg[_data.getIndex(rix)];
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], preAgg[_data.getIndex(i)]);
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {

		if(_colIndexes.length == 1)
			leftMultByMatrixNoPreAggSingleCol(matrix, result, rl, ru, cl, cu);
		else
			lmMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void leftMultByMatrixNoPreAggSingleCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl,
		int cu) {
		final double[] retV = result.getDenseBlockValues();
		final int nColM = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] dictVals = _dict.getValues(); // guaranteed dense double since we only have one column.

		if(matrix.isInSparseFormat())
			lmSparseMatrixNoPreAggSingleCol(matrix.getSparseBlock(), nColM, retV, nColRet, dictVals, rl, ru, cl, cu);
		else
			lmDenseMatrixNoPreAggSingleCol(matrix.getDenseBlockValues(), nColM, retV, nColRet, dictVals, rl, ru, cl, cu);

	}

	private void lmSparseMatrixNoPreAggSingleCol(SparseBlock sb, int nColM, double[] retV, int nColRet, double[] vals,
		int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes[0];

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			final double[] aval = sb.values(r);
			final int offR = r * nColRet;
			for(int i = apos; i < alen; i++)
				retV[offR + colOut] += aval[i] * vals[_data.getIndex(aix[i])];
		}
	}

	private void lmDenseMatrixNoPreAggSingleCol(double[] mV, int nColM, double[] retV, int nColRet, double[] vals,
		int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes[0];
		for(int r = rl; r < ru; r++) {
			final int offL = r * nColM;
			final int offR = r * nColRet;
			for(int c = cl; c < cu; c++)
				retV[offR + colOut] += mV[offL + c] * vals[_data.getIndex(r)];
		}
	}

	private void lmMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		if(matrix.isInSparseFormat())
			lmSparseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
		else
			lmDenseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		final SparseBlock sb = matrix.getSparseBlock();

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			final double[] aval = sb.values(r);
			final int offR = r * nColRet;
			for(int i = apos; i < alen; i++)
				_dict.multiplyScalar(aval[i], retV, offR, _data.getIndex(aix[i]), _colIndexes);
		}
	}

	private void lmDenseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final double[] retV = result.getDenseBlockValues();
		final int nColM = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] mV = matrix.getDenseBlockValues();
		for(int r = rl; r < ru; r++) {
			final int offL = r * nColM;
			final int offR = r * nColRet;
			for(int c = cl; c < cu; c++)
				_dict.multiplyScalar(mV[offL + c], retV, offR, _data.getIndex(c), _colIndexes);
		}

	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
		_data.preAggregateDense(m, preAgg, rl, ru, cl, cu);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru) {
		if(rl == ru - 1)
			for(int r = rl; r < ru; r++) {
				final int apos = sb.pos(r);
				final int alen = sb.size(r) + apos;
				final int[] aix = sb.indexes(r);
				final double[] avals = sb.values(r);
				for(int j = apos; j < alen; j++)
					preAgg[_data.getIndex(aix[j])] += avals[j];
			}
		else
			throw new NotImplementedException();
	}

	@Override
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		_data.preAggregateDDC_DDC(that._data, that._dict, ret, that._colIndexes.length);
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		_data.preAggregateDDC_SDCZ(that._data, that._dict, that._indexes, ret, that._colIndexes.length);
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AIterator itThat = that._indexes.getIterator();
		final int nCol = that._colIndexes.length;
		final int finalOff = that._indexes.getOffsetToLast();
		final double[] v = ret.getValues();
		while(true) {
			final int to = _data.getIndex(itThat.value());
			that._dict.addToEntry(v, 0, to, nCol);
			if(itThat.value() == finalOff)
				break;
			itThat.next();
		}
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupDDC && ((ColGroupDDC) that)._data == _data;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.DDC;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _data.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		if((op.fn instanceof Plus || op.fn instanceof Minus)
		// && _dict instanceof MatrixBlockDictionary &&
		// ((MatrixBlockDictionary) _dict).getMatrixBlock().isInSparseFormat()
		) {
			final double v0 = op.executeScalar(0);
			if(v0 == 0)
				return this;
			final double[] reference = FORUtil.createReference(_colIndexes.length, v0);
			return ColGroupDDCFOR.create(_colIndexes, _numRows, _dict, _data, getCachedCounts(), reference);
		}
		return create(_colIndexes, _numRows, _dict.applyScalarOp(op), _data, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		return create(_colIndexes, _numRows, _dict.applyUnaryOp(op), _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, _numRows, ret, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if((op.fn instanceof Plus || op.fn instanceof Minus)
		// && _dict instanceof MatrixBlockDictionary &&
		// ((MatrixBlockDictionary) _dict).getMatrixBlock().isInSparseFormat()
		) {
			final double[] reference = ColGroupUtils.binaryDefRowRight(op, v, _colIndexes);
			return ColGroupDDCFOR.create(_colIndexes, _numRows, _dict, _data, getCachedCounts(), reference);
		}
		final ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, _numRows, ret, _data, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_data = MapToFactory.readIn(in);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(nRows, nRows, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected int numRowsToMultiply() {
		return _numRows;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data);
		return sb.toString();
	}
}
