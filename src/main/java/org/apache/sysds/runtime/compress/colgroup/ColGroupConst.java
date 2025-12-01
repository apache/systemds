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
import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.AIdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetEmpty;
import org.apache.sysds.runtime.compress.colgroup.scheme.ConstScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class ColGroupConst extends ADictBasedColGroup implements IContainDefaultTuple, AOffsetsGroup, IMapToDataGroup {

	private static final long serialVersionUID = -7387793538322386611L;

	/**
	 * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
	 * 
	 * @param colIndices The Colum indexes for the column group.
	 * @param dict       The dictionary containing one tuple for the entire compression.
	 */
	private ColGroupConst(IColIndex colIndices, IDictionary dict) {
		super(colIndices, dict);
	}

	/**
	 * Create constructor for a ColGroup Const this constructor ensures that if the dictionary input is empty an Empty
	 * column group is constructed.
	 * 
	 * @param colIndices The column indexes in the column group
	 * @param dict       The dictionary to use
	 * @return A Colgroup either const or empty.
	 */
	public static AColGroup create(IColIndex colIndices, IDictionary dict) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
		else if(dict.getNumberOfValues(colIndices.size()) > 1 && !(dict instanceof PlaceHolderDict)) {
			// extract dict first row
			final double[] nd = new double[colIndices.size()];
			for(int i = 0; i < colIndices.size(); i++)
				nd[i] = dict.getValue(i);

			return ColGroupConst.create(colIndices, nd);
		}
		else
			return new ColGroupConst(colIndices, dict);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param values The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup create(double[] values) {
		return create(ColIndexFactory.create(values.length), values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * It is assumed that the column group is intended for use, therefore zero value is allowed.
	 * 
	 * @param cols  The specific column indexes that is contained in this constant group.
	 * @param value The value contained in all cells.
	 * @return A Constant column group.
	 */
	public static AColGroup create(IColIndex cols, double value) {
		if(cols.size() == 0)
			throw new DMLCompressionException("Invalid number of columns");
		else if(value == 0)
			return new ColGroupEmpty(cols);
		final int numCols = cols.size();
		double[] values = new double[numCols];
		for(int i = 0; i < numCols; i++)
			values[i] = value;
		return create(cols, values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param cols   The specific column indexes that is contained in this constant group.
	 * @param values The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup create(IColIndex cols, double[] values) {
		if(cols.size() != values.length)
			throw new DMLCompressionException("Invalid size of values compared to columns");
		boolean allZero = true;
		for(double d : values)
			if(d != 0.0) {
				allZero = false;
				break;
			}

		if(allZero)
			return new ColGroupEmpty(cols);
		else
			return ColGroupConst.create(cols, Dictionary.create(values));

	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns.
	 * @param dict    The dictionary to contain int the Constant group.
	 * @return A Constant column group.
	 */
	public static AColGroup create(int numCols, IDictionary dict) {
		if(dict instanceof MatrixBlockDictionary) {
			MatrixBlock mbd = ((MatrixBlockDictionary) dict).getMatrixBlock();
			if(mbd.getNumColumns() != numCols && mbd.getNumRows() != 1) {
				throw new DMLCompressionException(
					"Invalid construction of const column group with different number of columns in arguments");
			}
		}
		else if(numCols != dict.getValues().length)
			throw new DMLCompressionException(
				"Invalid construction of const column group with different number of columns in arguments");
		return ColGroupConst.create(ColIndexFactory.create(numCols), dict);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns
	 * @param value   The value contained in all cells.
	 * @return A Constant column group.
	 */
	public static AColGroup create(int numCols, double value) {
		if(numCols <= 0)
			throw new DMLCompressionException("Invalid construction of constant column group with cols: " + numCols);
		final IColIndex colIndices = ColIndexFactory.create(numCols);

		if(value == 0)
			return new ColGroupEmpty(colIndices);
		return ColGroupConst.create(colIndices, value);
	}

	/**
	 * Get dense values from colgroupConst.
	 * 
	 * @return the dictionary vector stored in this column group
	 */
	public double[] getValues() {
		double[] values;
		if(getDictionary() instanceof MatrixBlockDictionary) {
			LOG.warn("Inefficient get values for constant column group (but it is allowed)");
			final MatrixBlock mb = ((MatrixBlockDictionary) getDictionary()).getMatrixBlock();
			if(mb.isInSparseFormat()) {
				values = new double[mb.getNumColumns()];
				SparseBlock sb = mb.getSparseBlock();
				final int alen = sb.size(0);
				final double[] aval = sb.values(0);
				final int[] aix = sb.indexes(0);
				for(int j = 0; j < alen; j++)
					values[aix[j]] = aval[j];
			}
			else
				values = mb.getDenseBlockValues();
		}
		else
			values = _dict.getValues();
		return values;
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		double v = preAgg[0];
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], v);
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.CONST;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.CONST;
	}

	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		// guaranteed to be containing some values therefore no check for empty.
		final int apos = sb.pos(0);
		final int alen = sb.size(0);
		final int[] aix = sb.indexes(0);
		final double[] avals = sb.values(0);

		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = apos; j < alen; j++)
				c[off + _colIndexes.get(aix[j])] += avals[j];
		}
	}

	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		if(db.isContiguous() && _colIndexes.size() == db.getDim(1) && offC == 0)
			decompressToDenseBlockAllColumnsContiguous(db, rl + offR, ru + offR);
		else
			decompressToDenseBlockGeneric(db, rl, ru, offR, offC);
	}

	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final int apos = sb.pos(0);
		final int alen = sb.size(0);
		final int[] aix = sb.indexes(0);
		final double[] avals = sb.values(0);

		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			for(int j = apos; j < alen; j++)
				ret.append(offT, _colIndexes.get(aix[j]) + offC, avals[j]);

	}

	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.size();
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes.get(j) + offC, _dict.getValue(j));
	}

	private final void decompressToDenseBlockAllColumnsContiguous(final DenseBlock db, final int rl, final int ru) {
		final double[] c = db.values(0);
		final int nCol = _colIndexes.size();
		final double[] values = _dict.getValues();
		final int start = rl * nCol;
		final int end = ru * nCol;
		for(int i = start; i < end; i++)
			c[i] += values[i % nCol];
	}

	private void decompressToDenseBlockGeneric(DenseBlock db, int rl, int ru, int offR, int offC) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < _colIndexes.size(); j++)
				c[off + _colIndexes.get(j)] += _dict.getValue(j);
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(colIdx);
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return create(_colIndexes, _dict.applyScalarOp(op));
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		return create(_colIndexes, _dict.applyUnaryOp(op));
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		return create(_colIndexes, _dict.binOpLeft(op, v, _colIndexes));
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		return create(_colIndexes, _dict.binOpRight(op, v, _colIndexes));
	}

	/**
	 * Take the values in this constant column group and add to the given constV. This allows us to completely ignore
	 * this column group for future calculations.
	 * 
	 * @param constV The output columns.
	 */
	public final void addToCommon(double[] constV) {
		if(_dict instanceof AIdentityDictionary) {
			MatrixBlock mb = ((AIdentityDictionary) _dict).getMBDict().getMatrixBlock();
			if(mb.isInSparseFormat())
				addToCommonSparse(constV, mb.getSparseBlock());
			else
				addToCommonDense(constV, mb.getDenseBlockValues());
		}
		else if(_dict instanceof MatrixBlockDictionary) {
			MatrixBlock mb = ((MatrixBlockDictionary) _dict).getMatrixBlock();
			if(mb.isInSparseFormat())
				addToCommonSparse(constV, mb.getSparseBlock());
			else
				addToCommonDense(constV, mb.getDenseBlockValues());
		}
		else
			addToCommonDense(constV, _dict.getValues());
	}

	private final void addToCommonDense(double[] constV, double[] values) {
		for(int i = 0; i < _colIndexes.size(); i++)
			constV[_colIndexes.get(i)] += values[i];
	}

	private final void addToCommonSparse(double[] constV, SparseBlock sb) {

		final int alen = sb.size(0);
		final int[] aix = sb.indexes(0);
		final double[] aval = sb.values(0);
		for(int i = 0; i < alen; i++)
			constV[_colIndexes.get(aix[i])] += aval[i];

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
	protected void computeSum(double[] c, int nRows) {
		c[0] += _dict.sum(new int[] {nRows}, _colIndexes.size());
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		_dict.colSum(c, new int[] {nRows}, _colIndexes);
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSq(new int[] {nRows}, _colIndexes.size());
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict.colSumSq(c, new int[] {nRows}, _colIndexes);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		final double vals = preAgg[0];
		for(int rix = rl; rix < ru; rix++)
			c[rix] += vals;
	}

	@Override
	public int getNumValues() {
		return 1;
	}

	@Override
	public void tsmm(double[] result, int numColumns, int nRows) {
		tsmm(result, numColumns, new int[] {nRows}, _dict, _colIndexes);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		LOG.warn("Do not use leftMultByMatrixNoPreAgg on ColGroupConst");
		final double[] rowSum = (cl != 0 && cu != matrix.getNumColumns()) ? // do partial row sum if range is requested
			CLALibLeftMultBy.rowSum(matrix, rl, ru, cl, cu) : // partial row sum
			matrix.rowSum().getDenseBlockValues(); // full row sum

		leftMultByRowSum(rowSum, result, rl, ru);
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		LOG.warn("Should never call leftMultByMatrixByAColGroup on ColGroupConst");
		final double[] rowSum = new double[result.getNumRows()];
		lhs.computeColSums(rowSum, nRows);
		leftMultByRowSum(rowSum, result, 0, result.getNumRows());
	}

	private void leftMultByRowSum(double[] rowSum, MatrixBlock result, int rl, int ru) {
		if(_dict instanceof MatrixBlockDictionary) {
			MatrixBlock mb = ((MatrixBlockDictionary) _dict).getMatrixBlock();
			if(mb.isInSparseFormat())
				ColGroupUtils.outerProduct(rowSum, mb.getSparseBlock(), _colIndexes, result.getDenseBlockValues(),
					result.getNumColumns(), rl, ru);
			else
				ColGroupUtils.outerProduct(rowSum, _dict.getValues(), _colIndexes, result.getDenseBlockValues(),
					result.getNumColumns(), rl, ru);
		}
		else
			ColGroupUtils.outerProduct(rowSum, _dict.getValues(), _colIndexes, result.getDenseBlockValues(),
				result.getNumColumns(), rl, ru);

	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called");
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		IColIndex colIndexes = ColIndexFactory.create(1);
		double v = _dict.getValue(idx);
		if(v == 0)
			return new ColGroupEmpty(colIndexes);
		else {
			IDictionary retD = Dictionary.create(new double[] {_dict.getValue(idx)});
			return create(colIndexes, retD);
		}
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		IDictionary retD = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.size());
		return create(outputCols, retD);
	}

	@Override
	public boolean containsValue(double pattern) {
		return _dict.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		return _dict.getNumberNonZeros(new int[] {nRows}, _colIndexes.size());
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		IDictionary replaced = _dict.replace(pattern, replace, _colIndexes.size());
		return create(_colIndexes, replaced);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		_dict.product(c, new int[] {nRows}, _colIndexes.size());
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		final double v = preAgg[0];
		for(int rix = rl; rix < ru; rix++)
			c[rix] *= v;
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProduct(c, new int[] {nRows}, _colIndexes);
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDouble(_colIndexes.size());
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSq(_colIndexes.size());
	}

	@Override
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDouble(_colIndexes.size());
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRows(builtin, _colIndexes.size());
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		CM_COV_Object ret = new CM_COV_Object();
		op.fn.execute(ret, _dict.getValue(0), nRows);
		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		IDictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.size());
		if(d == null) {
			if(max <= 0)
				return null;
			return ColGroupEmpty.create(max);
		}
		else
			return create(max, d);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nCols = getNumCols();
		return e.getCost(nRows, 1, nCols, 1, 1.0);
	}

	protected AColGroup copyAndSet(IColIndex colIndexes, double[] newDictionary) {
		return create(colIndexes, Dictionary.create(newDictionary));
	}

	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, newDictionary);
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
		if(colIndexes != null && preAgg != null)
			return create(colIndexes, preAgg);
		else
			return null;
	}

	public static ColGroupConst read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		return new ColGroupConst(cols, dict);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		return this;
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupConst && g._colIndexes.size() == _colIndexes.size() &&
			((ColGroupConst) g)._dict.equals(_dict))
			return this;
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		for(int i = 0; i < g.length; i++) {
			final AColGroup gs = g[i];
			if(!_colIndexes.equals(gs._colIndexes))
				throw new DMLCompressionException("Invalid columns not matching " + gs._colIndexes + " " + _colIndexes);
			if(gs instanceof ColGroupConst) {
				if(this._dict.equals(((ColGroupConst) gs)._dict))
					continue; // common case
				else
					throw new NotImplementedException("Appending const not equivalent");
			}
			else if(gs instanceof ColGroupEmpty)
				throw new NotImplementedException("Appending empty and const");
			else
				return gs.appendNInternal(g, blen, rlen);
		}
		return this;
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return ConstScheme.create(this);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		EstimationFactors ef = new EstimationFactors(1, 1, 1, _dict.getSparsity());
		return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), CompressionType.CONST,
			getEncoding());
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(this);
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupEmpty || that instanceof ColGroupConst;
	}

	@Override
	public double[] getDefaultTuple() {
		return _dict.getValues();
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return ColGroupConst.create(newColIndex, _dict.reorder(reordering));
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Values: "));
		sb.append(_dict.getClass().getSimpleName());
		sb.append(_dict.getString(_colIndexes.size()));
		return sb.toString();
	}

	@Override
	public AOffset getOffsets() {
		return new OffsetEmpty();
	}

	@Override
	public AMapToData getMapToData() {
		return MapToFactory.create(0, 0);
	}

	@Override
	public double getSparsity() {
		return 1.0;
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		// guaranteed to be containing some values therefore no check for empty.
		final int apos = sb.pos(0);
		final int alen = sb.size(0);
		final int[] aix = sb.indexes(0);
		final double[] avals = sb.values(0);

		for(int j = apos; j < alen; j++) {
			final int rowOut = _colIndexes.get(aix[j]);
			final double[] c = db.values(rowOut);
			final int off = db.pos(rowOut); // row offset out.
			final double v = avals[j];
			for(int i = rl; i < ru; i++)
				c[off + i] += v;
		}
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		for(int j = 0; j < _colIndexes.size(); j++) {
			final int rowOut = _colIndexes.get(j);
			final double[] c = db.values(rowOut);
			final int off = db.pos(rowOut);
			double v = dict[j];
			for(int i = rl; i < ru; i++) {
				c[off + i] += v;
			}
		}
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock sb, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		if(!(right instanceof ColGroupConst))
			return super.combineWithSameIndex(nRow, nCol, right);
		final IColIndex combIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		final IDictionary b = ((ColGroupConst) right).getDictionary();
		final IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		return create(combIndex, combined);
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		final int s = _colIndexes.size();
		final int[] newColumns = new int[s * multiplier];
		final double[] newConst = new double[s * multiplier];
		final double[] vals = _dict.getValues();
		for(int i = 0; i < multiplier; i++) {
			for(int j = 0; j < s; j++)
				newColumns[i * s + j] = _colIndexes.get(j) + nColOrg * i;
			System.arraycopy(vals, 0, newConst, s * i, s);
		}
		return new AColGroup[] {create(ColIndexFactory.create(newColumns), newConst)};
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		for(int i = 0; i < right.size(); i++) {
			AColGroup g = right.get(i);

			if(!(g instanceof ColGroupConst) && !(g instanceof ColGroupEmpty)) {
				return super.combineWithSameIndex(nRow, nCol, right);
			}
		}

		IColIndex combinedIndex = _colIndexes;
		int i = 0;
		for(AColGroup g : right) {
			i += 1;
			combinedIndex = combinedIndex.combine(g.getColIndices().shift(nCol * i));
		}
		final IDictionary combined = combineDictionaries(nCol, right);

		return create(combinedIndex, combined);
	}

	@Override
	protected boolean allowShallowIdentityRightMult() {
		return true;
	}

	@Override
	public AColGroup sort() {
		return this;
	}

	@Override
	public AColGroup removeEmptyRows(boolean[] selectV, int rOut) {
		return this;
	}

	@Override
	protected AColGroup removeEmptyColsSubset(IColIndex newColumnIDs, IntArrayList selectedColumns) {
		return ColGroupConst.create(newColumnIDs, _dict.sliceColumns(selectedColumns, getNumCols()));
	}
}
