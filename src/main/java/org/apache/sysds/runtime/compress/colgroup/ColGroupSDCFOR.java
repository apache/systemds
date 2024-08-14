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
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.RemoveEmptyOffsetsTmp;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
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
 * ColGroup for Patched Frame Of Reference.
 * 
 * This column group fits perfectly into the collection of compression groups
 * 
 * It can be constructed when a SDCZeros group get a non zero default value. Then a natural extension is to transform
 * the group into a PFOR group, since the default value is then treated as an offset, and the dictionary can be copied
 * with no modifications.
 * 
 */
public class ColGroupSDCFOR extends ASDC implements IMapToDataGroup, IFrameOfReferenceGroup {

	private static final long serialVersionUID = 3883228464052204203L;

	/** Pointers to row indexes in the dictionary. */
	protected final AMapToData _data;

	/** Reference values in this column group */
	protected final double[] _reference;

	private ColGroupSDCFOR(IColIndex colIndices, int numRows, IDictionary dict, AOffset indexes, AMapToData data,
		int[] cachedCounts, double[] reference) {
		super(colIndices, numRows, dict, indexes, cachedCounts);
		// allow for now 1 data unique.
		_data = data;
		_reference = reference;
		if(CompressedMatrixBlock.debug) {

			if(data.getUnique() == 1)
				LOG.warn("SDCFor unique is 1, indicate it should have been SDCSingle please add support");
			else if(data.getUnique() != dict.getNumberOfValues(colIndices.size()))
				throw new DMLCompressionException("Invalid construction of SDCZero group");

			_data.verify();
			_indexes.verify(_data.size());
		}

	}

	public static AColGroup create(IColIndex colIndexes, int numRows, IDictionary dict, AOffset offsets, AMapToData data,
		int[] cachedCounts, double[] reference) {
		final boolean allZero = ColGroupUtils.allZero(reference);
		if(allZero && dict == null)
			return new ColGroupEmpty(colIndexes);
		else if(dict == null)
			return ColGroupConst.create(colIndexes, reference);
		else if(allZero)
			return ColGroupSDCZeros.create(colIndexes, numRows, dict, offsets, data, cachedCounts);
		// else if(data.getUnique() == 1){
		// TODO add support for changing to SDCSINGLE.
		// }
		return new ColGroupSDCFOR(colIndexes, numRows, dict, offsets, data, cachedCounts, reference);
	}

	public static AColGroup sparsifyFOR(ColGroupSDC g) {
		// subtract default.
		final double[] constV = g._defaultTuple;
		final AColGroupValue clg = (AColGroupValue) g.subtractDefaultTuple();
		return create(g.getColIndices(), g._numRows, clg._dict, g._indexes, g._data, g.getCachedCounts(), constV);
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDCFOR;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCFOR;
	}

	@Override
	public double[] getDefaultTuple() {
		return _reference;
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts);
	}

	@Override
	public AMapToData getMapToData() {
		return _data;
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowMxx(c, builtin, rl, ru, preAgg, _data, _indexes, _numRows, preAgg[preAgg.length - 1]);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it == null || it.value() != r)
			return _reference[colIdx];
		return _dict.getValue(_data.getIndex(it.getDataIndex()), colIdx, _colIndexes.size()) + _reference[colIdx];
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.executeScalar(_reference[i]);
		if(op.fn instanceof Plus || op.fn instanceof Minus)
			return create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final IDictionary newDict = _dict.applyScalarOp(op);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else {
			final IDictionary newDict = _dict.applyScalarOpWithReference(op, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newRef = ColGroupUtils.unaryOperator(op, _reference);
		final IDictionary newDict = _dict.applyUnaryOpWithReference(op, _reference, newRef);
		return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(v[_colIndexes.get(i)], _reference[i]);

		if(op.fn instanceof Plus || op.fn instanceof Minus) // only edit reference
			return create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final IDictionary newDict = _dict.binOpLeftWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(_reference[i], v[_colIndexes.get(i)]);

		if(op.fn instanceof Plus || op.fn instanceof Minus)// only edit reference
			return create(_colIndexes, _numRows, _dict, _indexes, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final IDictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final IDictionary newDict = _dict.binOpRightWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		_data.write(out);
		for(double d : _reference)
			out.writeDouble(d);
	}

	public static ColGroupSDCFOR read(DataInput in, int nRows) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AOffset indexes = OffsetFactory.readIn(in);
		AMapToData data = MapToFactory.readIn(in);
		double[] reference = ColGroupIO.readDoubleArray(cols.size(), in);
		return new ColGroupSDCFOR(cols, nRows, dict, indexes, data, null, reference);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		ret += 8 * _colIndexes.size(); // reference values.
		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		size += _data.getInMemorySize();
		size += 8 * _colIndexes.size();
		return size;
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

			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), nRef);
		}
		else
			return create(_colIndexes, _numRows, newDict, _indexes, _data, getCachedCounts(), _reference);

	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return _dict.aggregateWithReference(c, builtin, _reference, true);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateColsWithReference(c, builtin, _colIndexes, _reference, true);
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
		// square sum the dictionary.
		c[0] += _dict.sumSqWithReference(getCounts(), _reference);
		final double refSum = ColGroupUtils.refSumSq(_reference);
		// Square sum of the reference values only for the rows that is not represented in the Offsets.
		c[0] += refSum * (_numRows - _data.size());
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		// square sum the dictionary
		_dict.colSumSqWithReference(c, getCounts(), _colIndexes, _reference);
		// Square sum of the reference values only for the rows that is not represented in the Offsets.
		for(int i = 0; i < _colIndexes.size(); i++) // correct for the reference sum.
			c[_colIndexes.get(i)] += _reference[i] * _reference[i] * (_numRows - _data.size());
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
		final int count = _numRows - _data.size();
		_dict.productWithReference(c, getCounts(), _reference, count);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		ColGroupSDC.computeRowProduct(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProductWithReference(c, getCounts(), _colIndexes, _reference);
		final int count = _numRows - _data.size();

		for(int x = 0; x < _colIndexes.size(); x++) {
			double v = c[_colIndexes.get(x)];
			c[_colIndexes.get(x)] = v != 0 ? v * Math.pow(_reference[x], count) : 0;
		}
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		IDictionary retDict = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.size());
		final double[] newDef = new double[idEnd - idStart];
		for(int i = idStart, j = 0; i < idEnd; i++, j++)
			newDef[j] = _reference[i];
		return create(outputCols, _numRows, retDict, _indexes, _data, getCounts(), newDef);
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		final IColIndex retIndexes = ColIndexFactory.create(1);
		if(_colIndexes.size() == 1) // early abort, only single column already.
			return create(retIndexes, _numRows, _dict, _indexes, _data, getCounts(), _reference);
		final double[] newDef = new double[] {_reference[idx]};
		final IDictionary retDict = _dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.size());
		return create(retIndexes, _numRows, retDict, _indexes, _data, getCounts(), newDef);
	}

	@Override
	public boolean containsValue(double pattern) {
		if(Double.isNaN(pattern) || Double.isInfinite(pattern))
			return ColGroupUtils.containsInfOrNan(pattern, _reference) || _dict.containsValue(pattern);
		else {
			// if the value is in reference then return true.
			for(double v : _reference)
				if(v == pattern)
					return true;

			return _dict.containsValueWithReference(pattern, _reference);
		}
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		final int[] counts = getCounts();
		final int count = _numRows - _data.size();
		long c = _dict.getNumberNonZerosWithReference(counts, _reference, nRows);
		for(int x = 0; x < _colIndexes.size(); x++)
			c += _reference[x] != 0 ? count : 0;
		return c;
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.size(); i++)
			constV[_colIndexes.get(i)] += _reference[i];
		return ColGroupSDCZeros.create(_colIndexes, _numRows, _dict, _indexes, _data, getCounts());
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		IDictionary d = _dict.rexpandColsWithReference(max, ignore, cast, (int) _reference[0]);
		return ColGroupSDC.rexpandCols(max, ignore, cast, nRows, d, _indexes, _data, getCachedCounts(),
			(int) _reference[0], _dict.getNumberOfValues(1));
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		// should be guaranteed to be one column therefore only one reference value.
		return _dict.centralMomentWithReference(op.fn, getCounts(), _reference[0], nRows);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = _data.size();
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	public double[] getCommon() {
		return _reference;
	}

	@Override
	protected AColGroup allocateRightMultiplicationCommon(double[] common, IColIndex colIndexes, IDictionary preAgg) {
		return create(colIndexes, _numRows, preAgg, _indexes, _data, getCachedCounts(), common);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return ColGroupConst.create(_colIndexes, Dictionary.create(_reference));
		AMapToData newData = _data.slice(off.lIndex, off.uIndex);
		return create(_colIndexes, ru - rl, _dict, off.offsetSlice, newData, null, _reference);
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _indexes, _data, getCachedCounts(), _reference);
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		for(int i = 1; i < g.length; i++) {
			if(!_colIndexes.equals(g[i]._colIndexes)) {
				LOG.warn("Not same columns therefore not appending \n" + _colIndexes + "\n\n" + g[i]._colIndexes);
				return null;
			}

			if(!(g[i] instanceof ColGroupSDCFOR)) {
				LOG.warn("Not SDCFOR but " + g[i].getClass().getSimpleName());
				return null;
			}

			final ColGroupSDCFOR gc = (ColGroupSDCFOR) g[i];
			if(!gc._dict.equals(_dict)) {
				LOG.warn("Not same Dictionaries therefore not appending \n" + _dict + "\n\n" + gc._dict);
				return null;
			}
		}
		AMapToData nd = _data.appendN(Arrays.copyOf(g, g.length, IMapToDataGroup[].class));
		AOffset no = _indexes.appendN(Arrays.copyOf(g, g.length, AOffsetsGroup[].class), getNumRows());
		return create(_colIndexes, rlen, _dict, no, nd, null, _reference);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(_data, _indexes, _numRows);
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that instanceof ColGroupSDCFOR) {
			ColGroupSDCFOR th = (ColGroupSDCFOR) that;
			return th._indexes == _indexes && th._data == _data;
		}
		else
			return false;
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new NotImplementedException();
	}

	@Override
	public int getNumberOffsets() {
		return _data.size();
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		throw new NotImplementedException();
	}

	@Override
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {

		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		final double[] combinedDefaultTuple = IContainDefaultTuple.combineDefaultTuples(_reference, right);

		return new ColGroupSDCFOR(combinedColIndex, this.getNumRows(), combined, _indexes, _data, getCachedCounts(),
			combinedDefaultTuple);
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		ColGroupSDCFOR rightSDC = ((ColGroupSDCFOR) right);
		IDictionary b = rightSDC.getDictionary();
		IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		double[] combinedDefaultTuple = new double[_reference.length + rightSDC._reference.length];
		System.arraycopy(_reference, 0, combinedDefaultTuple, 0, _reference.length);
		System.arraycopy(rightSDC._reference, 0, combinedDefaultTuple, _reference.length, rightSDC._reference.length);

		return new ColGroupSDCFOR(combinedColIndex, this.getNumRows(), combined, _indexes, _data, getCachedCounts(),
			combinedDefaultTuple);
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		IntArrayList[] splitOffs = new IntArrayList[multiplier];
		IntArrayList[] tmpMaps = new IntArrayList[multiplier];
		for(int i = 0; i < multiplier; i++) {
			splitOffs[i] = new IntArrayList();
			tmpMaps[i] = new IntArrayList();
		}

		AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();

		while(it.value() != last) {
			final int v = it.value(); // offset
			final int d = it.getDataIndex(); // data index value
			final int m = _data.getIndex(d);

			final int outV = v / multiplier;
			final int outM = v % multiplier;

			tmpMaps[outM].appendValue(m);
			splitOffs[outM].appendValue(outV);

			it.next();
		}

		// last value
		final int v = it.value();
		final int d = it.getDataIndex();
		final int m = _data.getIndex(d);
		final int outV = v / multiplier;
		final int outM = v % multiplier;
		tmpMaps[outM].appendValue(m);
		splitOffs[outM].appendValue(outV);

		// iterate through all rows.

		AOffset[] offs = new AOffset[multiplier];
		AMapToData[] maps = new AMapToData[multiplier];
		for(int i = 0; i < multiplier; i++) {
			offs[i] = OffsetFactory.createOffset(splitOffs[i]);
			maps[i] = MapToFactory.create(_data.getUnique(), tmpMaps[i]);
		}

		// assign columns
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _numRows / multiplier, _dict, offs[i], maps[i], null, _reference);
		}
		return res;
	}

	@Override
	public AColGroup sort() {
		if(getNumCols() > 1)
			throw new NotImplementedException();
		// TODO restore support for run length encoding.

		final int[] counts = getCounts();
		// get the sort index
		final int[] r = _dict.sort();

		// find default value position.
		// todo use binary search for minor improvements.
		int defIdx = counts.length;
		for(int i = 0; i < r.length; i++) {
			if(_dict.getValue(r[i], 0, 1) >= 0) {
				defIdx = i;
				break;
			}
		}

		int nondefault = _data.size();
		int defaultLength = _numRows - nondefault;
		AMapToData m = MapToFactory.create(nondefault, counts.length);
		int[] offsets = new int[nondefault];

		int off = 0;
		for(int i = 0; i < counts.length; i++) {
			if(i < defIdx) {
				for(int j = 0; j < counts[r[i]]; j++) {
					offsets[off] = off;
					m.set(off++, r[i]);
				}
			}
			else {// if( i >= defIdx){
				for(int j = 0; j < counts[r[i]]; j++) {
					offsets[off] = off + defaultLength;
					m.set(off++, r[i]);
				}
			}
		}

		AOffset o = OffsetFactory.createOffset(offsets);
		return ColGroupSDCFOR.create(_colIndexes, _numRows, _dict, o, m, counts, _reference);
	}

	@Override
	public AColGroup removeEmptyRows(boolean[] selectV, int rOut) {
		final RemoveEmptyOffsetsTmp offsetTmp = _indexes.removeEmptyRows(selectV, rOut);
		final AMapToData nm = _data.removeEmpty(offsetTmp.select);
		return ColGroupSDCFOR.create(_colIndexes, rOut, _dict, offsetTmp.retOffset, nm, null, _reference);
	}

	@Override
	protected AColGroup removeEmptyColsSubset(IColIndex newColumnIDs, IntArrayList selectedColumns) {
		double[] ref = new double[selectedColumns.size()];
		for(int i = 0; i < selectedColumns.size(); i++) {
			ref[i] = _reference[selectedColumns.get(i)];
		}
		return ColGroupSDCFOR.create(newColumnIDs, _numRows, _dict.sliceColumns(selectedColumns, getNumCols()), _indexes, _data, null,
			ref);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data);
		sb.append(String.format("\n%15s", "Reference:"));
		sb.append(Arrays.toString(_reference));
		return sb.toString();
	}

}
