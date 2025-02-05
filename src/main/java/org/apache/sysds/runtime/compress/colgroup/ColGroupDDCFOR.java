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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 */
public class ColGroupDDCFOR extends AMorphingMMColGroup implements IFrameOfReferenceGroup {
	private static final long serialVersionUID = -5769772089913918987L;

	/** Pointers to row indexes in the dictionary */
	protected final AMapToData _data;

	/** Reference values in this column group */
	protected final double[] _reference;

	private ColGroupDDCFOR(IColIndex colIndexes, IDictionary dict, double[] reference, AMapToData data,
		int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);
		_data = data;
		_reference = reference;
	}

	public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts,
		double[] reference) {
		final boolean allZero = ColGroupUtils.allZero(reference);
		if(dict == null && allZero)
			return new ColGroupEmpty(colIndexes);
		else if(dict == null)
			return ColGroupConst.create(colIndexes, reference);
		else if(data.getUnique() == 1)
			return ColGroupConst.create(colIndexes,
				dict.binOpRight(new BinaryOperator(Plus.getPlusFnObject()), reference));
		else if(allZero)
			return ColGroupDDC.create(colIndexes, dict, data, cachedCounts);
		else
			return new ColGroupDDCFOR(colIndexes, dict, reference, data, cachedCounts);
	}

	public static AColGroup sparsifyFOR(ColGroupDDC g) {
		// It is assumed whoever call this does not use an empty Dictionary in g.
		final int nCol = g.getColIndices().size();
		final MatrixBlockDictionary mbd = g._dict.getMBDict(nCol);
		if(mbd != null) {

			final MatrixBlock mb = mbd.getMatrixBlock();

			final double[] ref = ColGroupUtils.extractMostCommonValueInColumns(mb);
			if(ref != null) {
				MatrixBlockDictionary mDict = mbd.binOpRight(new BinaryOperator(Minus.getMinusFnObject()), ref);
				return create(g.getColIndices(), mDict, g._data, g.getCachedCounts(), ref);
			}
			else
				return g;
		}
		else {
			throw new NotImplementedException("The dictionary was empty... highly unlikely");
		}
	}

	public CompressionType getCompType() {
		return CompressionType.DDCFOR;
	}

	@Override
	public double[] getDefaultTuple() {
		return _reference;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r), colIdx, _colIndexes.size()) + _reference[colIdx];
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
	public ColGroupType getColGroupType() {
		return ColGroupType.DDCFOR;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _data.getInMemorySize();
		size += 8 * _colIndexes.size();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.executeScalar(_reference[i]);
		if(op.fn instanceof Plus || op.fn instanceof Minus)
			return create(_colIndexes, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final IDictionary newDict = _dict.applyScalarOp(op);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else {
			final IDictionary newDict = _dict.applyScalarOpWithReference(op, _reference, newRef);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newRef = ColGroupUtils.unaryOperator(op, _reference);
		final IDictionary newDict = _dict.applyUnaryOpWithReference(op, _reference, newRef);
		return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(v[_colIndexes.get(i)], _reference[i]);

		if(op.fn instanceof Plus || op.fn instanceof Minus) // only edit reference
			return create(_colIndexes, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final IDictionary newDict = _dict.binOpLeftWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(_reference[i], v[_colIndexes.get(i)]);

		if(op.fn instanceof Plus || op.fn instanceof Minus)// only edit reference
			return create(_colIndexes, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final IDictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final IDictionary newDict = _dict.binOpRightWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
		for(double d : _reference)
			out.writeDouble(d);
	}

	public static ColGroupDDCFOR read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AMapToData data = MapToFactory.readIn(in);
		double[] ref = ColGroupIO.readDoubleArray(cols.size(), in);
		return new ColGroupDDCFOR(cols, dict, ref, data, null);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += 8 * _colIndexes.size(); // reference values.
		return ret;
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(nRows, nRows, nCols, nVals, _dict.getSparsity());
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		final IDictionary newDict = _dict.replaceWithReference(pattern, replace, _reference);
		boolean patternInReference = false;
		for(double d : _reference)
			if(pattern == d) {
				patternInReference = true;
				break;
			}
		if(patternInReference) {
			double[] nRef = new double[_reference.length];
			for(int i = 0; i < _reference.length; i++)
				if(Util.eq(pattern, _reference[i]))
					nRef[i] = replace;
				else
					nRef[i] = _reference[i];

			return create(_colIndexes, newDict, _data, getCachedCounts(), nRef);
		}
		else
			return create(_colIndexes, newDict, _data, getCachedCounts(), _reference);

	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return _dict.aggregateWithReference(c, builtin, _reference, false);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateColsWithReference(c, builtin, _colIndexes, _reference, false);
	}

	@Override
	protected void computeSum(double[] c, int nRows) {
		// trick, use normal sum
		super.computeSum(c, nRows);
		// and add all sum of reference multiplied with nrows.
		final double refSum = ColGroupUtils.refSum(_reference);
		c[0] += refSum * nRows;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		// trick, use normal sum
		super.computeColSums(c, nRows);
		// and add reference multiplied with number of rows.
		for(int i = 0; i < _colIndexes.size(); i++)
			c[_colIndexes.get(i)] += _reference[i] * nRows;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSqWithReference(getCounts(), _reference);
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict.colSumSqWithReference(c, getCounts(), _colIndexes, _reference);
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDoubleWithReference(_reference);
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSqWithReference(_reference);
	}

	@Override
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDoubleWithReference(_reference);
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRowsWithReference(builtin, _reference);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		_dict.productWithReference(c, getCounts(), _reference, 0);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++)
			c[rix] *= preAgg[_data.getIndex(rix)];
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProductWithReference(c, getCounts(), _colIndexes, _reference);
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		IDictionary retDict = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.size());
		final double[] newDef = new double[idEnd - idStart];
		for(int i = idStart, j = 0; i < idEnd; i++, j++)
			newDef[j] = _reference[i];
		return create(outputCols, retDict, _data, getCounts(), newDef);
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		final IColIndex retIndexes = ColIndexFactory.create(1);
		if(_colIndexes.size() == 1) // early abort, only single column already.
			return create(retIndexes, _dict, _data, getCounts(), _reference);
		final double[] newDef = new double[] {_reference[idx]};
		final IDictionary retDict = _dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.size());
		return create(retIndexes, retDict, _data, getCounts(), newDef);

	}

	@Override
	public boolean containsValue(double pattern) {
		if(Double.isNaN(pattern) || Double.isInfinite(pattern))
			return ColGroupUtils.containsInfOrNan(pattern, _reference) || _dict.containsValue(pattern);
		else
			return _dict.containsValueWithReference(pattern, _reference);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		// to be safe just assume the worst fully dense for DDCFOR
		return (long) _colIndexes.size() * nRows;
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.size(); i++)
			constV[_colIndexes.get(i)] += _reference[i];
		return ColGroupDDC.create(_colIndexes, _dict, _data, getCounts());
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		final int def = (int) _reference[0];
		IDictionary d = _dict.rexpandColsWithReference(max, ignore, cast, def);
		if(d == null) {
			if(max <= 0)
				return null;
			return ColGroupEmpty.create(max);
		}
		else {
			IColIndex outCols = ColIndexFactory.create(d.getNumberOfColumns(_dict.getNumberOfValues(1)));
			return ColGroupDDC.create(outCols, d, _data, getCachedCounts());
		}
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		// should be guaranteed to be one column therefore only one reference value.
		CM_COV_Object ret = _dict.centralMomentWithReference(op.fn, getCounts(), _reference[0], nRows);
		return ret;
	}

	@Override
	public double[] getCommon() {
		return _reference;
	}

	@Override
	protected AColGroup allocateRightMultiplicationCommon(double[] common, IColIndex colIndexes, IDictionary preAgg) {
		return create(colIndexes, preAgg, _data, getCachedCounts(), common);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		AMapToData sliceMap = _data.slice(rl, ru);
		return new ColGroupDDCFOR(_colIndexes, _dict, _reference, sliceMap, null);
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, newDictionary, _data, getCachedCounts(), _reference);
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupDDCFOR && g.getColIndices().equals(_colIndexes)) {
			ColGroupDDCFOR gDDC = (ColGroupDDCFOR) g;
			if(Arrays.equals(_reference, gDDC._reference) && gDDC._dict.equals(_dict)) {
				AMapToData nd = _data.append(gDDC._data);
				return create(_colIndexes, _dict, nd, null, _reference);
			}
		}
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		throw new NotImplementedException();
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup recompress() {
		throw new NotImplementedException();
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		IEncode enc = getEncoding();
		EstimationFactors ef = new EstimationFactors(getNumValues(), _data.size(), _data.size(), _dict.getSparsity());
		return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), getCompType(), enc);
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(_data);
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupDDCFOR && ((ColGroupDDCFOR) that)._data == _data;
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new NotImplementedException();
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			decompressToSparseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;

			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos];
			decompressToDenseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		final double[] combinedReference = IContainDefaultTuple.combineDefaultTuples(_reference, right);

		return ColGroupDDCFOR.create(combinedColIndex, combined, _data, getCachedCounts(), combinedReference);
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		IDictionary b = ((ColGroupDDCFOR) right).getDictionary();
		IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		double[] combinedReference = new double[_reference.length + right.getNumCols()];
		System.arraycopy(_reference, 0, combinedReference, 0, _reference.length);
		double[] rightReference = ((ColGroupDDCFOR) right).getDefaultTuple();
		System.arraycopy(rightReference, 0, combinedReference, _reference.length, rightReference.length);

		return ColGroupDDCFOR.create(combinedColIndex, combined, _data, getCachedCounts(), combinedReference);
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		AMapToData[] maps = _data.splitReshapeDDC(multiplier);
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _dict, maps[i], null, _reference);
		}

		return res;
	}

	@Override
	protected boolean allowShallowIdentityRightMult() {
		return false;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Data: "));
		sb.append(_data);
		sb.append(String.format("\n%15s", "Reference:"));
		sb.append(Arrays.toString(_reference));
		return sb.toString();
	}
}
