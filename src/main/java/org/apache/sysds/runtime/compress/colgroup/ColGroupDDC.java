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
import java.util.concurrent.ExecutorService;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.DDCScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.jboss.netty.handler.codec.compression.CompressionException;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 */
public class ColGroupDDC extends APreAgg implements IMapToDataGroup {
	private static final long serialVersionUID = -5769772089913918987L;

	protected final AMapToData _data;

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	private ColGroupDDC(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);
		_data = data;

		if(CompressedMatrixBlock.debug) {
			if(getNumValues() == 0)
				throw new DMLCompressionException("Invalid construction with empty dictionary");
			if(data.size() == 0)
				throw new DMLCompressionException("Invalid length of the data. is zero");

			if(data.getUnique() != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException("Invalid map to dict Map has:" + data.getUnique() + " while dict has "
					+ dict.getNumberOfValues(colIndexes.size()));
			int[] c = getCounts();
			if(c.length != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException("Invalid DDC Construction");
			data.verify();
		}
	}

	public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
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
			_colIndexes.decompressToDenseFromSparse(sb, vr, off, c);
		}
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int idxSize = _colIndexes.size();
		if(db.isContiguous()) {
			final int nColOut = db.getDim(1);
			if(idxSize == 1 && nColOut == 1)
				decompressToDenseBlockDenseDictSingleColOutContiguous(db, rl, ru, offR, offC, values);
			else if(idxSize == 1)
				decompressToDenseBlockDenseDictSingleColContiguous(db, rl, ru, offR, offC, values);
			else if(idxSize == nColOut) // offC == 0 implied
				decompressToDenseBlockDenseDictAllColumnsContiguous(db, rl, ru, offR, values, idxSize);
			else if(offC == 0 && offR == 0)
				decompressToDenseBlockDenseDictNoOff(db, rl, ru, values);
			else if(offC == 0)
				decompressToDenseBlockDenseDictNoColOffset(db, rl, ru, offR, values, idxSize, nColOut);
			else
				decompressToDenseBlockDenseDictGeneric(db, rl, ru, offR, offC, values, idxSize);
		}
		else
			decompressToDenseBlockDenseDictGeneric(db, rl, ru, offR, offC, values, idxSize);
	}

	private final void decompressToDenseBlockDenseDictSingleColContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values) {
		final double[] c = db.values(0);
		final int nCols = db.getDim(1);
		final int colOff = _colIndexes.get(0) + offC;
		for(int i = rl, offT = (rl + offR) * nCols + colOff; i < ru; i++, offT += nCols)
			c[offT] += values[_data.getIndex(i)];

	}

	@Override
	public AMapToData getMapToData() {
		return _data;
	}

	private final void decompressToDenseBlockDenseDictSingleColOutContiguous(DenseBlock db, int rl, int ru, int offR,
		int offC, double[] values) {
		final double[] c = db.values(0);
		decompressToDenseBlockDenseDictSingleColOutContiguous(c, rl, ru, offR + _colIndexes.get(0), values, _data);
	}

	private final static void decompressToDenseBlockDenseDictSingleColOutContiguous(double[] c, int rl, int ru, int offR,
		double[] values, AMapToData data) {
		data.decompressToRange(c, rl, ru, offR, values);

	}

	private final void decompressToDenseBlockDenseDictAllColumnsContiguous(DenseBlock db, int rl, int ru, int offR,
		double[] values, int nCol) {
		final double[] c = db.values(0);
		for(int r = rl; r < ru; r++) {
			final int start = _data.getIndex(r) * nCol;
			final int offStart = (offR + r) * nCol;
			LibMatrixMult.vectAdd(values, c, start, offStart, nCol);
		}
	}

	private final void decompressToDenseBlockDenseDictNoColOffset(DenseBlock db, int rl, int ru, int offR,
		double[] values, int nCol, int colOut) {
		int off = (rl + offR) * colOut;
		for(int i = rl, offT = rl + offR; i < ru; i++, off += colOut) {
			final double[] c = db.values(offT);
			final int rowIndex = _data.getIndex(i) * nCol;
			_colIndexes.decompressVec(nCol, c, off, values, rowIndex);
		}
	}

	private final void decompressToDenseBlockDenseDictNoOff(DenseBlock db, int rl, int ru, double[] values) {
		final int nCol = _colIndexes.size();
		final int nColU = db.getDim(1);
		final double[] c = db.values(0);
		for(int i = rl; i < ru; i++) {
			final int off = i * nColU;
			final int rowIndex = _data.getIndex(i) * nCol;
			_colIndexes.decompressVec(nCol, c, off, values, rowIndex);
		}
	}

	private final void decompressToDenseBlockDenseDictGeneric(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values, int nCol) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			final int rowIndex = _data.getIndex(i) * nCol;
			_colIndexes.decompressVec(nCol, c, off, values, rowIndex);
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
				ret.append(offT, offC + _colIndexes.get(aix[j]), aval[j]);
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		decompressToSparseBlockDenseDictionary(ret, rl, ru, offR, offC, values, _colIndexes.size());
	}

	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values, int nCol) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final int rowIndex = _data.getIndex(i) * nCol;
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes.get(j) + offC, values[rowIndex + j]);
		}
	}

	@Override
	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		for(int i = rl; i < ru; i++) {
			final int vr = _data.getIndex(i);
			if(sb.isEmpty(vr))
				continue;
			final int apos = sb.pos(vr);
			final int alen = sb.size(vr) + apos;
			final int[] aix = sb.indexes(vr);
			final double[] aval = sb.values(vr);
			for(int j = apos; j < alen; j++) {
				final int rowOut = _colIndexes.get(aix[j]);
				final double[] c = db.values(rowOut);
				final int off = db.pos(rowOut);
				c[off + i] += aval[j];
			}
		}
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		final int nCol = _colIndexes.size();
		for(int j = 0; j < nCol; j++) {
			final int rowOut = _colIndexes.get(j);
			final double[] c = db.values(rowOut);
			final int off = db.pos(rowOut);
			for(int i = rl; i < ru; i++) {
				final double v = dict[_data.getIndex(i) * nCol + j];
				c[off + i] += v;
			}
		}
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR sbr, SparseBlock sb, int nColOut) {

		int[] colCounts = _dict.countNNZZeroColumns(getCounts());
		for(int j = 0; j < _colIndexes.size(); j++)
			sbr.allocate(_colIndexes.get(j), colCounts[j]);

		for(int i = 0; i < _data.size(); i++) {
			int di = _data.getIndex(i);
			if(sb.isEmpty(di))
				continue;

			final int apos = sb.pos(di);
			final int alen = sb.size(di) + apos;
			final int[] aix = sb.indexes(di);
			final double[] aval = sb.values(di);

			for(int j = apos; j < alen; j++) {
				sbr.append(_colIndexes.get(aix[j]), i, aval[apos]);
			}
		}

	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR sbr, double[] dict, int nColOut) {
		int[] colCounts = _dict.countNNZZeroColumns(getCounts());
		for(int j = 0; j < _colIndexes.size(); j++)
			sbr.allocate(_colIndexes.get(j), colCounts[j]);

		final int nCol = _colIndexes.size();
		for(int j = 0; j < nCol; j++) {
			final int rowOut = _colIndexes.get(j);
			SparseRow r = sbr.get(rowOut);

			for(int i = 0; i < _data.size(); i++) {
				final double v = dict[_data.getIndex(i) * nCol + j];
				r = r.append(i, v);
			}
			sbr.set(rowOut, r, false);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r), colIdx, _colIndexes.size());
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
		if(_colIndexes.size() == 1)
			leftMultByMatrixNoPreAggSingleCol(matrix, result, rl, ru, cl, cu);
		else
			lmMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void leftMultByMatrixNoPreAggSingleCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl,
		int cu) {
		final DenseBlock retV = result.getDenseBlock();
		final int nColM = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] dictVals = _dict.getValues(); // guaranteed dense double since we only have one column.
		if(matrix.isEmpty())
			return;
		else if(matrix.isInSparseFormat()) {
			if(cl != 0 || cu != _data.size())
				lmSparseMatrixNoPreAggSingleCol(matrix.getSparseBlock(), nColM, retV, nColRet, dictVals, rl, ru, cl, cu);
			else
				lmSparseMatrixNoPreAggSingleCol(matrix.getSparseBlock(), nColM, retV, nColRet, dictVals, rl, ru);
		}
		else if(!matrix.getDenseBlock().isContiguous())
			lmDenseMatrixNoPreAggSingleColNonContiguous(matrix.getDenseBlock(), nColM, retV, nColRet, dictVals, rl, ru, cl,
				cu);
		else
			lmDenseMatrixNoPreAggSingleCol(matrix.getDenseBlockValues(), nColM, retV, nColRet, dictVals, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggSingleCol(SparseBlock sb, int nColM, DenseBlock retV, int nColRet, double[] vals,
		int rl, int ru) {

		if(retV.isContiguous())
			lmSparseMatrixNoPreAggSingleColContiguous(sb, nColM, retV.valuesAt(0), nColRet, vals, rl, ru);
		else
			lmSparseMatrixNoPreAggSingleColGeneric(sb, nColM, retV, nColRet, vals, rl, ru);
	}

	private void lmSparseMatrixNoPreAggSingleColGeneric(SparseBlock sb, int nColM, DenseBlock ret, int nColRet,
		double[] vals, int rl, int ru) {
		final int colOut = _colIndexes.get(0);

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			final double[] aval = sb.values(r);
			final int offR = ret.pos(r);
			final double[] retV = ret.values(r);

			for(int i = apos; i < alen; i++)
				retV[offR + colOut] += aval[i] * vals[_data.getIndex(aix[i])];
		}
	}

	private void lmSparseMatrixNoPreAggSingleColContiguous(SparseBlock sb, int nColM, double[] retV, int nColRet,
		double[] vals, int rl, int ru) {
		final int colOut = _colIndexes.get(0);

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

	private void lmSparseMatrixNoPreAggSingleCol(SparseBlock sb, int nColM, DenseBlock retV, int nColRet, double[] vals,
		int rl, int ru, int cl, int cu) {
		if(retV.isContiguous())
			lmSparseMatrixNoPreAggSingleColContiguous(sb, nColM, retV.valuesAt(0), nColRet, vals, rl, ru, cl, cu);
		else
			lmSparseMatrixNoPreAggSingleColGeneric(sb, nColM, retV, nColRet, vals, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggSingleColGeneric(SparseBlock sb, int nColM, DenseBlock ret, int nColRet,
		double[] vals, int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes.get(0);

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int aposSkip = sb.posFIndexGTE(r, cl);
			final int[] aix = sb.indexes(r);
			if(aposSkip <= -1 || aix[apos + aposSkip] >= cu)
				continue;
			final int alen = sb.size(r) + apos;
			final double[] aval = sb.values(r);
			final int offR = ret.pos(r);
			final double[] retV = ret.values(r);
			// final int offR = r * nColRet;
			for(int i = apos + aposSkip; i < alen && aix[i] < cu; i++)
				retV[offR + colOut] += aval[i] * vals[_data.getIndex(aix[i])];
		}
	}

	private void lmSparseMatrixNoPreAggSingleColContiguous(SparseBlock sb, int nColM, double[] retV, int nColRet,
		double[] vals, int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes.get(0);

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int aposSkip = sb.posFIndexGTE(r, cl);
			final int[] aix = sb.indexes(r);
			if(aposSkip <= -1 || aix[apos + aposSkip] >= cu)
				continue;
			final int alen = sb.size(r) + apos;
			final double[] aval = sb.values(r);
			final int offR = r * nColRet;
			for(int i = apos + aposSkip; i < alen && aix[i] < cu; i++)
				retV[offR + colOut] += aval[i] * vals[_data.getIndex(aix[i])];
		}
	}

	private void lmDenseMatrixNoPreAggSingleColNonContiguous(DenseBlock db, int nColM, DenseBlock retV, int nColRet,
		double[] vals, int rl, int ru, int cl, int cu) {
		lmDenseMatrixNoPreAggSingleColNonContiguousInGeneric(db, nColM, retV, nColRet, vals, rl, ru, cl, cu);
	}

	private void lmDenseMatrixNoPreAggSingleCol(double[] mV, int nColM, DenseBlock retV, int nColRet, double[] vals,
		int rl, int ru, int cl, int cu) {
		if(retV.isContiguous())
			lmDenseMatrixNoPreAggSingleColContiguous(mV, nColM, retV.valuesAt(0), nColRet, vals, rl, ru, cl, cu);
		else
			lmDenseMatrixNoPreAggSingleColGeneric(mV, nColM, retV, nColRet, vals, rl, ru, cl, cu);
	}

	private void lmDenseMatrixNoPreAggSingleColNonContiguousInGeneric(DenseBlock db, int nColM, DenseBlock ret,
		int nColRet, double[] vals, int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes.get(0);
		for(int r = rl; r < ru; r++) {
			final int offL = db.pos(r);
			final double[] mV = db.values(r);
			final int offR = ret.pos(r);
			final double[] retV = ret.values(r);
			for(int c = cl; c < cu; c++)
				retV[offR + colOut] += mV[offL + c] * vals[_data.getIndex(c)];
		}
	}

	private void lmDenseMatrixNoPreAggSingleColGeneric(double[] mV, int nColM, DenseBlock ret, int nColRet,
		double[] vals, int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes.get(0);
		for(int r = rl; r < ru; r++) {
			final int offL = r * nColM;
			final int offR = ret.pos(r);
			final double[] retV = ret.values(r);
			for(int c = cl; c < cu; c++)
				retV[offR + colOut] += mV[offL + c] * vals[_data.getIndex(c)];
		}
	}

	private void lmDenseMatrixNoPreAggSingleColContiguous(double[] mV, int nColM, double[] retV, int nColRet,
		double[] vals, int rl, int ru, int cl, int cu) {
		final int colOut = _colIndexes.get(0);
		for(int r = rl; r < ru; r++) {
			final int offL = r * nColM;
			final int offR = r * nColRet;
			for(int c = cl; c < cu; c++)
				retV[offR + colOut] += mV[offL + c] * vals[_data.getIndex(c)];
		}
	}

	private void lmMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {

		if(matrix.isInSparseFormat())
			lmSparseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
		else
			lmDenseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final DenseBlock db = result.getDenseBlock();
		final SparseBlock sb = matrix.getSparseBlock();

		if(cl != 0 || cu != _data.size()) {
			// sub part
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final double[] retV = db.values(r);
				final int pos = db.pos(r);
				lmSparseMatrixRowColRange(sb, r, pos, retV, cl, cu);
			}
		}
		else {
			for(int r = rl; r < ru; r++)
				_data.lmSparseMatrixRow(sb, r, db, _colIndexes, _dict);
		}
	}

	private final void lmSparseMatrixRowColRange(SparseBlock sb, int r, int offR, double[] retV, int cl, int cu) {
		final int apos = sb.pos(r);
		final int aposSkip = sb.posFIndexGTE(r, cl);
		final int[] aix = sb.indexes(r);
		if(aposSkip <= -1 || aix[apos + aposSkip] >= cu)
			return;
		final int alen = sb.size(r) + apos;
		final double[] aval = sb.values(r);
		for(int i = apos + aposSkip; i < alen && aix[i] < cu; i++)
			_dict.multiplyScalar(aval[i], retV, offR, _data.getIndex(aix[i]), _colIndexes);
	}

	private void lmDenseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		if(matrix.getDenseBlock().isContiguous())
			lmDenseMatrixNoPreAggMultiColContiguous(matrix, result, rl, ru, cl, cu);
		else
			lmDenseMatrixNoPreAggMultiColNonContiguous(matrix.getDenseBlock(), result, rl, ru, cl, cu);
	}

	private void lmDenseMatrixNoPreAggMultiColContiguous(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl,
		int cu) {
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

	private void lmDenseMatrixNoPreAggMultiColNonContiguous(DenseBlock db, MatrixBlock result, int rl, int ru, int cl,
		int cu) {
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		for(int r = rl; r < ru; r++) {
			final int offL = db.pos(r);
			final double[] mV = db.values(r);
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
	public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {
		DenseBlock db = that.getDenseBlock();
		DenseBlock retDB = ret.getDenseBlock();
		for(int i = rl; i < ru; i++)
			leftMMIdentityPreAggregateDenseSingleRow(db.values(i), db.pos(i), retDB.values(i), retDB.pos(i), cl, cu);
	}

	@Override
	public void rightDecompressingMult(MatrixBlock right, MatrixBlock ret, int rl, int ru, int nRows, int crl, int cru) {
		if(_dict instanceof IdentityDictionary)
			identityRightDecompressingMult(right, ret, rl, ru, crl, cru);
		else
			defaultRightDecompressingMult(right, ret, rl, ru, crl, cru);
	}

	private void identityRightDecompressingMult(MatrixBlock right, MatrixBlock ret, int rl, int ru, int crl, int cru) {
		final double[] b = right.getDenseBlockValues();
		final double[] c = ret.getDenseBlockValues();
		final int jd = right.getNumColumns();
		final DoubleVector vVec = DoubleVector.zero(SPECIES);
		final int vLen = SPECIES.length();
		final int lenJ = cru - crl;
		final int end = cru - (lenJ % vLen);
		for(int i = rl; i < ru; i++) {
			int k = _data.getIndex(i);
			final int offOut = i * jd + crl;
			final double aa = 1;
			final int k_right = _colIndexes.get(k);
			vectMM(aa, b, c, end, jd, crl, cru, offOut, k_right, vLen, vVec);
		}
	}

	private void defaultRightDecompressingMult(MatrixBlock right, MatrixBlock ret, int rl, int ru, int crl, int cru) {
		final double[] a = _dict.getValues();
		final double[] b = right.getDenseBlockValues();
		final double[] c = ret.getDenseBlockValues();
		final int kd = _colIndexes.size();
		final int jd = right.getNumColumns();
		final DoubleVector vVec = DoubleVector.zero(SPECIES);
		final int vLen = SPECIES.length();

		final int blkzI = 32;
		final int blkzK = 24;
		final int lenJ = cru - crl;
		final int end = cru - (lenJ % vLen);
		for(int bi = rl; bi < ru; bi += blkzI) {
			final int bie = Math.min(ru, bi + blkzI);
			for(int bk = 0; bk < kd; bk += blkzK) {
				final int bke = Math.min(kd, bk + blkzK);
				for(int i = bi; i < bie; i++) {
					int offi = _data.getIndex(i) * kd;
					final int offOut = i * jd + crl;
					for(int k = bk; k < bke; k++) {
						final double aa = a[offi + k];
						final int k_right = _colIndexes.get(k);
						vectMM(aa, b, c, end, jd, crl, cru, offOut, k_right, vLen, vVec);
					}
				}
			}
		}
	}

	final void vectMM(double aa, double[] b, double[] c, int endT, int jd, int crl, int cru, int offOut, int k, int vLen, DoubleVector vVec) {
		vVec = vVec.broadcast(aa);
		final int offj = k * jd;
		final int end = endT + offj;
		for(int j = offj + crl; j < end; j += vLen, offOut += vLen) {
			DoubleVector res = DoubleVector.fromArray(SPECIES, c, offOut);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, j);
			res = vVec.fma(bVec, res);
			res.intoArray(c, offOut);
		}
		for(int j = end; j < cru + offj; j++, offOut++) {
			double bb = b[j];
			c[offOut] += bb * aa;
		}
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {
		if(cl != 0 || cu != _data.size()) {
			throw new NotImplementedException();
		}
		_data.preAggregateSparse(sb, preAgg, rl, ru);
	}

	@Override
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		try {

			_data.preAggregateDDC_DDC(that._data, that._dict, ret, that._colIndexes.size());
		}
		catch(Exception e) {
			throw new CompressionException(that.toString(), e);
		}
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		_data.preAggregateDDC_SDCZ(that._data, that._dict, that._indexes, ret, that._colIndexes.size());
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final AOffsetIterator itThat = that._indexes.getOffsetIterator();
		final int nCol = that._colIndexes.size();
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
		_data.preAggregateDDC_RLE(that._ptr, that._data, that._dict, ret, that._colIndexes.size());
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
			final double[] reference = ColGroupUtils.createReference(_colIndexes.size(), v0);
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
		IDictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, ret, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if((op.fn instanceof Plus || op.fn instanceof Minus) && _dict instanceof MatrixBlockDictionary &&
			((MatrixBlockDictionary) _dict).getMatrixBlock().isInSparseFormat()) {
			final double[] reference = ColGroupUtils.binaryDefRowRight(op, v, _colIndexes);
			return ColGroupDDCFOR.create(_colIndexes, _dict, _data, getCachedCounts(), reference);
		}
		final IDictionary ret;
		if(_colIndexes.size() == 1)
			ret = _dict.applyScalarOp(new RightScalarOperator(op.fn, v[_colIndexes.get(0)]));
		else
			ret = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, ret, _data, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
	}

	public static ColGroupDDC read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AMapToData data = MapToFactory.readIn(in);
		return new ColGroupDDC(cols, dict, data, null);
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
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
		if(preAgg != null)
			return create(colIndexes, preAgg, _data, getCachedCounts());
		else
			return null;
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		try {
			return ColGroupDDC.create(_colIndexes, _dict, _data.slice(rl, ru), null);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to slice out sub part DDC: " + rl + " " + ru, e);
		}
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, newDictionary, _data, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupDDC) {
			if(g.getColIndices().equals(_colIndexes)) {

				ColGroupDDC gDDC = (ColGroupDDC) g;
				if(gDDC._dict.equals(_dict)) {
					AMapToData nd = _data.append(gDDC._data);
					return create(_colIndexes, _dict, nd, null);
				}
				else
					LOG.warn("Not same Dictionaries therefore not appending DDC\n" + _dict + "\n\n" + gDDC._dict);
			}
			else
				LOG.warn("Not same columns therefore not appending DDC\n" + _colIndexes + "\n\n" + g.getColIndices());
		}
		else
			LOG.warn("Not DDC but " + g.getClass().getSimpleName() + ", therefore not appending DDC");
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		for(int i = 1; i < g.length; i++) {
			if(!_colIndexes.equals(g[i]._colIndexes)) {
				LOG.warn("Not same columns therefore not appending DDC\n" + _colIndexes + "\n\n" + g[i]._colIndexes);
				return null;
			}

			if(!(g[i] instanceof ColGroupDDC)) {
				LOG.warn("Not DDC but " + g[i].getClass().getSimpleName() + ", therefore not appending DDC");
				return null;
			}

			final ColGroupDDC gDDC = (ColGroupDDC) g[i];
			if(!gDDC._dict.equals(_dict)) {
				LOG.warn("Not same Dictionaries therefore not appending DDC\n" + _dict + "\n\n" + gDDC._dict);
				return null;
			}
		}
		AMapToData nd = _data.appendN(Arrays.copyOf(g, g.length, IMapToDataGroup[].class));
		return create(_colIndexes, _dict, nd, null);
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return DDCScheme.create(this);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		try {

			IEncode enc = getEncoding();
			EstimationFactors ef = new EstimationFactors(_data.getUnique(), _data.size(), _data.size(),
				_dict.getSparsity());
			return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), getCompType(), enc);
		}
		catch(Exception e) {
			throw new DMLCompressionException(this.toString(), e);
		}
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(_data);
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return ColGroupDDC.create(newColIndex, _dict.reorder(reordering), _data, getCachedCounts());
	}

	@Override
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos]; // column index with 1
			decompressToSparseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		// morph(CompressionType.UNCOMPRESSED, _data.size()).sparseSelection(selection, ret, rl, ru);;
		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos]; // column index with 1
			decompressToDenseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	private void leftMMIdentityPreAggregateDenseSingleRow(double[] values, int pos, double[] values2, int pos2, int cl,
		int cu) {
		IdentityDictionary a = (IdentityDictionary) _dict;
		if(_colIndexes instanceof RangeIndex)
			leftMMIdentityPreAggregateDenseSingleRowRangeIndex(values, pos, values2, pos2, cl, cu);
		else {

			pos += cl; // left side matrix position offset.
			if(a.withEmpty()) {
				final int nVal = _dict.getNumberOfValues(_colIndexes.size()) - 1;
				for(int rc = cl; rc < cu; rc++, pos++) {
					final int idx = _data.getIndex(rc);
					if(idx != nVal)
						values2[pos2 + _colIndexes.get(idx)] += values[pos];
				}
			}
			else {
				for(int rc = cl; rc < cu; rc++, pos++)
					values2[pos2 + _colIndexes.get(_data.getIndex(rc))] += values[pos];
			}
		}
	}

	private void leftMMIdentityPreAggregateDenseSingleRowRangeIndex(double[] values, int pos, double[] values2, int pos2,
		int cl, int cu) {
		IdentityDictionary a = (IdentityDictionary) _dict;

		final int firstCol = pos2 + _colIndexes.get(0);
		pos += cl; // left side matrix position offset.
		if(a.withEmpty()) {
			final int nVal = _dict.getNumberOfValues(_colIndexes.size()) - 1;
			for(int rc = cl; rc < cu; rc++, pos++) {
				final int idx = _data.getIndex(rc);
				if(idx != nVal)
					values2[firstCol + idx] += values[pos];
			}
		}
		else {
			for(int rc = cl; rc < cu; rc++, pos++)
				values2[firstCol + _data.getIndex(rc)] += values[pos];
		}
	}

	@Override
	public AColGroup morph(CompressionType ct, int nRow) {
		// return this;
		if(ct == getCompType())
			return this;
		else if(ct == CompressionType.SDC) {
			// return this;
			int[] counts = getCounts();
			int maxId = maxIndex(counts);
			double[] def = _dict.getRow(maxId, _colIndexes.size());

			int offsetSize = nRow - counts[maxId];
			int[] offsets = new int[offsetSize];
			AMapToData reducedData = MapToFactory.create(offsetSize, _data.getUnique());
			int o = 0;
			for(int i = 0; i < nRow; i++) {
				int v = _data.getIndex(i);
				if(v != maxId) {
					offsets[o] = i;
					reducedData.set(o, v);
					o++;
				}
			}

			return ColGroupSDC.create(_colIndexes, _data.size(), _dict, def, OffsetFactory.createOffset(offsets),
				reducedData, null);
		}
		else if(ct == CompressionType.CONST) {
			// if(1 < getNumValues()) {
			String thisS = this.toString();
			if(thisS.length() > 10000)
				thisS = thisS.substring(0, 10000) + "...";
			LOG.warn("Tried to morph to const from DDC but impossible: " + thisS);
			return this;
			// }
		}
		else if(ct == CompressionType.DDCFOR)
			return this; // it does not make sense to change to FOR.
		else
			return super.morph(ct, nRow);
	}

	private static int maxIndex(int[] counts) {
		int id = 0;
		for(int i = 1; i < counts.length; i++) {
			if(counts[i] > counts[id]) {
				id = i;
			}
		}
		return id;
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		return new ColGroupDDC(combinedColIndex, combined, _data, getCachedCounts());
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		IDictionary b = ((ColGroupDDC) right).getDictionary();
		IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		return new ColGroupDDC(combinedColIndex, combined, _data, getCachedCounts());
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		AMapToData[] maps = _data.splitReshapeDDC(multiplier);
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _dict, maps[i], null);
		}
		return res;
	}

	@Override
	public AColGroup[] splitReshapePushDown(int multiplier, int nRow, int nColOrg, ExecutorService pool)
		throws Exception {
		AMapToData[] maps = _data.splitReshapeDDCPushDown(multiplier, pool);
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _dict, maps[i], null);
		}
		return res;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data);
		return sb.toString();
	}

	@Override
	protected boolean allowShallowIdentityRightMult() {
		return true;
	}

}
