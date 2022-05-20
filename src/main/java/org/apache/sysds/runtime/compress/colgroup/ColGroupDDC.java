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
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
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

	/** Constructor for serialization */
	protected ColGroupDDC() {
		super();
	}

	private ColGroupDDC(int[] colIndexes, ADictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);
		if(data.getUnique() != dict.getNumberOfValues(colIndexes.length))
			throw new DMLCompressionException("Invalid construction of DDC group " + data.getUnique() + " vs. "
				+ dict.getNumberOfValues(colIndexes.length));
		_data = data;
	}

	public static AColGroup create(int[] colIndexes, ADictionary dict, AMapToData data, int[] cachedCounts) {
		if(data.getUnique() == 1)
			return ColGroupConst.create(colIndexes, dict);
		else if(dict == null)
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupDDC(colIndexes, dict, data, cachedCounts);
	}

	public AColGroup sparsifyFOR() {
		return ColGroupDDCFOR.sparsifyFOR(this);
	}

	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		for(int r = rl, offT = rl + offR; r < ru; r++, offT++) {
			final int vr = _data.getIndex(r);
			if(sb.isEmpty(vr))
				continue;
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int apos = sb.pos(vr);
			final int alen = sb.size(vr) + apos;
			final int[] aix = sb.indexes(vr);
			final double[] aval = sb.values(vr);
			for(int j = apos; j < alen; j++)
				c[off + _colIndexes[aix[j]]] += aval[j];
		}
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		if(db.isContiguous()) {
			if(_colIndexes.length == 1 && db.getDim(1) == 1)
				decompressToDenseBlockDenseDictSingleColOutContiguous(db, rl, ru, offR, offC, values);
			else if(_colIndexes.length == 1)
				decompressToDenseBlockDenseDictSingleColContiguous(db, rl, ru, offR, offC, values);
			else if(_colIndexes.length == db.getDim(1)) // offC == 0 implied
				decompressToDenseBlockDenseDictAllColumnsContiguous(db, rl, ru, offR, values);
			else if(offC == 0 && offR == 0)
				decompressToDenseBlockDenseDictNoOff(db, rl, ru, values);
			else if(offC == 0)
				decompressToDenseBlockDenseDictNoColOffset(db, rl, ru, offR, values);
			else
				decompressToDenseBlockDenseDictGeneric(db, rl, ru, offR, offC, values);
		}
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

	private void decompressToDenseBlockDenseDictNoColOffset(DenseBlock db, int rl, int ru, int offR, double[] values) {
		final int nCol = _colIndexes.length;
		final int colOut = db.getDim(1);
		int off = (rl + offR) * colOut;
		for(int i = rl, offT = rl + offR; i < ru; i++, off += colOut) {
			final double[] c = db.values(offT);
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[rowIndex + j];
		}
	}

	private void decompressToDenseBlockDenseDictNoOff(DenseBlock db, int rl, int ru, double[] values) {
		final int nCol = _colIndexes.length;
		final int nColU = db.getDim(1);
		final double[] c = db.values(0);
		for(int i = rl; i < ru; i++) {
			final int off = i * nColU;
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += values[rowIndex + j];
		}
	}

	private void decompressToDenseBlockDenseDictGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
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
		for(int r = rl, offT = rl + offR; r < ru; r++, offT++) {
			final int vr = _data.getIndex(r);
			if(sb.isEmpty(vr))
				continue;
			final int apos = sb.pos(vr);
			final int alen = sb.size(vr) + apos;
			final int[] aix = sb.indexes(vr);
			final double[] aval = sb.values(vr);
			for(int j = apos; j < alen; j++)
				ret.append(offT, offC + _colIndexes[aix[j]], aval[j]);
		}
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
		return _dict.getValue(_data.getIndex(r), colIdx, _colIndexes.length);
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
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] *= preAgg[_data.getIndex(rix)];
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

		if(matrix.isInSparseFormat()) {
			if(cl != 0 || cu != _data.size())
				throw new NotImplementedException();
			lmSparseMatrixNoPreAggSingleCol(matrix.getSparseBlock(), nColM, retV, nColRet, dictVals, rl, ru);
		}
		else
			lmDenseMatrixNoPreAggSingleCol(matrix.getDenseBlockValues(), nColM, retV, nColRet, dictVals, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggSingleCol(SparseBlock sb, int nColM, double[] retV, int nColRet, double[] vals,
		int rl, int ru) {
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
				retV[offR + colOut] += mV[offL + c] * vals[_data.getIndex(c)];
		}
	}

	private void lmMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		if(matrix.isInSparseFormat()) {
			if(cl != 0 || cu != _data.size())
				throw new NotImplementedException(
					"Not implemented left multiplication on sparse without it being entire input");
			lmSparseMatrixNoPreAggMultiCol(matrix, result, rl, ru);
		}
		else
			lmDenseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
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
		_data.preAggregateSparse(sb, preAgg, rl, ru);
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
		final AOffsetIterator itThat = that._indexes.getOffsetIterator();
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
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		_data.preAggregateDDC_RLE(that._ptr, that._data, that._dict, ret, that._colIndexes.length);
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
		if((op.fn instanceof Plus || op.fn instanceof Minus)) {
			final double v0 = op.executeScalar(0);
			if(v0 == 0)
				return this;
			final double[] reference = ColGroupUtils.createReference(_colIndexes.length, v0);
			return ColGroupDDCFOR.create(_colIndexes, _dict, _data, getCachedCounts(), reference);
		}
		return create(_colIndexes, _dict.applyScalarOp(op), _data, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		return create(_colIndexes, _dict.applyUnaryOp(op), _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		ADictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, ret, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if((op.fn instanceof Plus || op.fn instanceof Minus) && _dict instanceof MatrixBlockDictionary &&
			((MatrixBlockDictionary) _dict).getMatrixBlock().isInSparseFormat()) {
			final double[] reference = ColGroupUtils.binaryDefRowRight(op, v, _colIndexes);
			return ColGroupDDCFOR.create(_colIndexes, _dict, _data, getCachedCounts(), reference);
		}
		final ADictionary ret = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, ret, _data, getCachedCounts());
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
		return _data.size();
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return _dict.aggregate(c, builtin);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		return _dict.containsValue(pattern);
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, int[] colIndexes, ADictionary preAgg) {
		if(preAgg != null)
			return create(colIndexes, preAgg, _data, getCachedCounts());
		else
			return null;
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
