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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToZero;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetEmpty;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes. If the values are very
 * sparse then the most common one is zero.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public class ColGroupSDCSingle extends ASDC {
	private static final long serialVersionUID = 3883228464052204200L;

	/** The default value stored in this column group */
	protected final double[] _defaultTuple;

	private ColGroupSDCSingle(IColIndex colIndices, int numRows, IDictionary dict, double[] defaultTuple,
		AOffset offsets, int[] cachedCounts) {
		super(colIndices, numRows, dict == null ? Dictionary.createNoCheck(new double[colIndices.size()]) : dict, offsets,
			cachedCounts);
		_defaultTuple = defaultTuple;
		if(CompressedMatrixBlock.debug) {
			_indexes.verify(_indexes.getSize());
		}
	}

	public static AColGroup create(IColIndex colIndexes, int numRows, IDictionary dict, double[] defaultTuple,
		AOffset offsets, int[] cachedCounts) {

		if(offsets instanceof OffsetEmpty)
			return ColGroupConst.create(colIndexes, defaultTuple);
		final boolean allZero = ColGroupUtils.allZero(defaultTuple);
		if(dict == null && allZero)
			return new ColGroupEmpty(colIndexes);
		else if(dict == null && offsets.getSize() * 2 > numRows + 2) {
			AOffset rev = offsets.reverse(numRows);
			return ColGroupSDCSingleZeros.create(colIndexes, numRows, Dictionary.create(defaultTuple), rev, cachedCounts);
		}
		else if(dict == null)
			return new ColGroupSDCSingle(colIndexes, numRows, null, defaultTuple, offsets, cachedCounts);
		else if(allZero)
			return ColGroupSDCSingleZeros.create(colIndexes, numRows, dict, offsets, cachedCounts);
		else if(offsets.getSize() * 2 > numRows + 2 && !(dict instanceof PlaceHolderDict)) {
			AOffset rev = offsets.reverse(numRows);
			return new ColGroupSDCSingle(colIndexes, numRows, Dictionary.create(defaultTuple), dict.getValues(), rev,
				null);
		}
		else
			return new ColGroupSDCSingle(colIndexes, numRows, dict, defaultTuple, offsets, cachedCounts);

	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDCSingle;
	}

	@Override
	public double[] getDefaultTuple() {
		return _defaultTuple;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it == null || it.value() != r)
			return _defaultTuple[colIdx];
		else
			return _dict.getValue(colIdx);
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDoubleWithDefault(_defaultTuple);
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSqWithDefault(_defaultTuple);
	}

	@Override
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDoubleWithDefault(_defaultTuple);
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRowsWithDefault(builtin, _defaultTuple);
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		double ret = _dict.aggregate(c, builtin);
		for(int i = 0; i < _defaultTuple.length; i++)
			ret = builtin.execute(ret, _defaultTuple[i]);
		return ret;
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateCols(c, builtin, _colIndexes);
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] = builtin.execute(c[_colIndexes.get(x)], _defaultTuple[x]);
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		computeRowSums(c, rl, ru, preAgg, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AOffset indexes, int nRows) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		final double def = preAgg[1];
		final double norm = preAgg[0];
		if(it != null) {

			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxOff = indexes.getOffsetToLast();
				while(true) {
					if(it.value() == r) {
						c[r] += norm;
						if(it.value() < maxOff)
							it.next();
						else {
							r++;
							break;
						}
					}
					else
						c[r] += def;
					r++;
				}
			}
			else {
				while(r < ru) {
					if(it.value() == r) {
						c[r++] += norm;
						it.next();
					}
					else
						c[r++] += def;
				}
				indexes.cacheIterator(it, ru);
			}
		}

		while(r < ru)
			c[r++] += def;

	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		computeRowMxx(c, builtin, rl, ru, _indexes, _numRows, preAgg[1], preAgg[0]);
	}

	protected static final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, AOffset indexes, int nRows,
		double def, double norm) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		if(it != null) {

			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxOff = indexes.getOffsetToLast();
				while(true) {
					if(it.value() == r) {
						c[r] = builtin.execute(c[r], norm);
						if(it.value() < maxOff)
							it.next();
						else {
							r++;
							break;
						}
					}
					else
						c[r] = builtin.execute(c[r], def);
					r++;
				}
			}
			else {
				while(r < ru) {
					if(it.value() == r) {
						c[r] = builtin.execute(c[r], norm);
						it.next();
					}
					else
						c[r] = builtin.execute(c[r], def);
					r++;
				}
				indexes.cacheIterator(it, ru);
			}
		}

		while(r < ru) {
			c[r] = builtin.execute(c[r], def);
			r++;
		}
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		computeRowProduct(c, rl, ru, _indexes, _numRows, preAgg[1], preAgg[0]);
	}

	protected static final void computeRowProduct(double[] c, int rl, int ru, AOffset indexes, int nRows, double def,
		double norm) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		if(it != null) {
			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxOff = indexes.getOffsetToLast();
				while(true) {
					if(it.value() == r) {
						c[r] *= norm;
						if(it.value() < maxOff)
							it.next();
						else {
							r++;
							break;
						}
					}
					else
						c[r] *= def;
					r++;
				}
			}
			else {
				while(r < ru) {
					if(it.value() == r) {
						c[r++] *= norm;
						it.next();
					}
					else
						c[r++] *= def;
				}
				indexes.cacheIterator(it, ru);
			}
		}

		while(r < ru)
			c[r++] *= def;
	}

	@Override
	protected void computeSum(double[] c, int nRows) {
		super.computeSum(c, nRows);
		int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _defaultTuple.length; x++)
			c[0] += _defaultTuple[x] * count;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		super.computeColSums(c, nRows);
		int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] += _defaultTuple[x] * count;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		super.computeSumSq(c, nRows);
		int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[0] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		super.computeColSumsSq(c, nRows);
		int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		final int count = _numRows - _indexes.getSize();
		_dict.productWithDefault(c, getCounts(), _defaultTuple, count);
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		super.computeColProduct(c, nRows);
		int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _colIndexes.size(); x++) {
			double v = c[_colIndexes.get(x)];
			c[_colIndexes.get(x)] = v != 0 ? v * Math.pow(_defaultTuple[x], count) : 0;
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		counts[0] = _indexes.getSize();
		return counts;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _indexes.getInMemorySize();
		size += 8 * _colIndexes.size();
		return size;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.executeScalar(_defaultTuple[i]);
		final IDictionary nDict = _dict.applyScalarOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i]);
		final IDictionary nDict = _dict.applyUnaryOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(v[_colIndexes.get(i)], _defaultTuple[i]);
		final IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i], v[_colIndexes.get(i)]);
		final IDictionary newDict = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		for(double d : _defaultTuple)
			out.writeDouble(d);
	}

	public static ColGroupSDCSingle read(DataInput in, int nRows) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AOffset indexes = OffsetFactory.readIn(in);
		double[] defaultTuple = ColGroupIO.readDoubleArray(cols.size(), in);
		return new ColGroupSDCSingle(cols, nRows, dict, defaultTuple, indexes, null);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _indexes.getExactSizeOnDisk();
		ret += 8 * _colIndexes.size(); // _default tuple values.
		return ret;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		IDictionary replaced = _dict.replace(pattern, replace, _colIndexes.size());
		double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = Util.eq(_defaultTuple[i], pattern) ? replace : _defaultTuple[i];

		return create(_colIndexes, _numRows, replaced, newDefaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.size(); i++)
			constV[_colIndexes.get(i)] += _defaultTuple[i];

		IDictionary subtractedDict = _dict.subtractTuple(_defaultTuple);
		return ColGroupSDCSingleZeros.create(_colIndexes, _numRows, subtractedDict, _indexes, getCachedCounts());
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		long nnz = super.getNumberNonZeros(nRows);
		final int count = _numRows - _indexes.getSize();
		for(int x = 0; x < _colIndexes.size(); x++)
			nnz += _defaultTuple[x] != 0 ? count : 0;
		return nnz;
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		return _dict.centralMomentWithDefault(op.fn, getCounts(), _defaultTuple[0], nRows);
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		IDictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.size());
		final int def = (int) _defaultTuple[0];
		if(d == null) {
			if(def <= 0){
				if(max > 0)
					return ColGroupEmpty.create(max);
				else 
					return null;
			}
			else if(def > max && max > 0)
				return ColGroupEmpty.create(max);
			else if(max <= 0)
				return null;
			else {
				double[] retDef = new double[max];
				retDef[((int) _defaultTuple[0]) - 1] = 1;
				return ColGroupSDCSingle.create(ColIndexFactory.create(max), nRows, null, retDef, _indexes, null);
			}
		}
		else {
			final IColIndex outCols = ColIndexFactory.create(d.getNumberOfColumns(_dict.getNumberOfValues(1)));
			if(def <= 0) {
				if(ignore)
					return ColGroupSDCSingleZeros.create(outCols, nRows, d, _indexes, getCachedCounts());
				else
					throw new DMLRuntimeException("Invalid content of zero in rexpand");
			}
			else if(def > max)
				return ColGroupSDCSingleZeros.create(outCols, nRows, d, _indexes, getCachedCounts());
			else {
				double[] retDef = new double[max];
				retDef[((int) _defaultTuple[0]) - 1] = 1;
				return ColGroupSDCSingle.create(outCols, nRows, d, retDef, _indexes, getCachedCounts());
			}
		}
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = _indexes.getSize();
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		final AColGroup ret = super.sliceMultiColumns(idStart, idEnd, outputCols);
		final double[] defTuple = new double[idEnd - idStart];
		for(int i = idStart, j = 0; i < idEnd; i++, j++)
			defTuple[j] = _defaultTuple[i];
		if(ret instanceof ColGroupEmpty)
			return create(ret._colIndexes, _numRows, null, defTuple, _indexes, null);
		else {
			ColGroupSDCSingle retSDC = (ColGroupSDCSingle) ret;
			return create(retSDC._colIndexes, _numRows, retSDC._dict, defTuple, _indexes, null);
		}
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		AColGroup ret = super.sliceSingleColumn(idx);
		double[] defTuple = new double[1];
		defTuple[0] = _defaultTuple[idx];
		if(ret instanceof ColGroupEmpty)
			return create(ret._colIndexes, _numRows, null, defTuple, _indexes, null);
		else {
			ColGroupSDCSingle retSDC = (ColGroupSDCSingle) ret;
			return create(retSDC._colIndexes, _numRows, retSDC._dict, defTuple, _indexes, null);
		}
	}

	@Override
	public boolean containsValue(double pattern) {
		if(_dict.containsValue(pattern))
			return true;
		else {
			for(double v : _defaultTuple)
				if(v == pattern)
					return true;
			return false;
		}
	}

	@Override
	public double[] getCommon() {
		return _defaultTuple;
	}

	@Override
	protected AColGroup allocateRightMultiplicationCommon(double[] common, IColIndex colIndexes, IDictionary preAgg) {
		return create(colIndexes, _numRows, preAgg, common, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return ColGroupConst.create(_colIndexes, Dictionary.create(_defaultTuple));
		return create(_colIndexes, ru - rl, _dict, _defaultTuple, off.offsetSlice, null);
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _defaultTuple, _indexes, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		for(int i = 1; i < g.length; i++) {
			final AColGroup gs = g[i];
			if(!_colIndexes.equals(gs._colIndexes))
				throw new DMLCompressionException(
					"Not same columns therefore not appending \n" + _colIndexes + "\n\n" + gs._colIndexes);

			if(!(gs instanceof AOffsetsGroup))
				throw new DMLCompressionException("Not SDC but " + gs.getClass().getSimpleName());

			if(gs instanceof ColGroupSDC) {
				final ColGroupSDC gc = (ColGroupSDC) gs;
				if(!gc._dict.equals(_dict))
					throw new DMLCompressionException(
						"Not same Dictionaries therefore not appending \n" + _dict + "\n\n" + gc._dict);
			}
			else if(gs instanceof ColGroupConst) {
				final ColGroupConst gc = (ColGroupConst) gs;
				if(!(gc._dict instanceof PlaceHolderDict) && gc._dict.equals(_defaultTuple))
					throw new DMLCompressionException("Not same default values therefore not appending:\n" + gc._dict
						+ "\n\n" + Arrays.toString(_defaultTuple));
			}
		}
		AOffset no = _indexes.appendN(Arrays.copyOf(g, g.length, AOffsetsGroup[].class), getNumRows());
		return create(_colIndexes, rlen, _dict, _defaultTuple, no, null);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(new MapToZero(getCounts()[0]), _indexes, _numRows);
	}

	@Override
	public int getNumberOffsets() {
		return getCounts()[0];
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that instanceof ColGroupSDCSingleZeros) {
			ColGroupSDCSingleZeros th = (ColGroupSDCSingleZeros) that;
			return th._indexes == _indexes;
		}
		else if(that instanceof ColGroupSDCSingle) {
			ColGroupSDCSingle th = (ColGroupSDCSingle) that;
			return th._indexes == _indexes;
		}
		else
			return false;
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return ColGroupSDCSingle.create(newColIndex, getNumRows(), _dict.reorder(reordering),
			ColGroupUtils.reorderDefault(_defaultTuple, reordering), _indexes, getCachedCounts());
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		ColGroupSDCSingle rightSDC = ((ColGroupSDCSingle) right);
		IDictionary b = rightSDC.getDictionary();
		IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
		IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
		double[] combinedDefaultTuple = new double[_defaultTuple.length + rightSDC._defaultTuple.length];
		System.arraycopy(_defaultTuple, 0, combinedDefaultTuple, 0, _defaultTuple.length);
		System.arraycopy(rightSDC._defaultTuple, 0, combinedDefaultTuple, _defaultTuple.length,
			rightSDC._defaultTuple.length);

		return new ColGroupSDCSingle(combinedColIndex, this.getNumRows(), combined, combinedDefaultTuple, _indexes,
			getCachedCounts());
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		final double[] combinedDefaultTuple = IContainDefaultTuple.combineDefaultTuples(_defaultTuple, right);

		// return new ColGroupDDC(combinedColIndex, combined, _data, getCachedCounts());
		return new ColGroupSDCSingle(combinedColIndex, this.getNumRows(), combined, combinedDefaultTuple, _indexes,
			getCachedCounts());
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		IntArrayList[] splitOffs = new IntArrayList[multiplier];
		for(int i = 0; i < multiplier; i++)
			splitOffs[i] = new IntArrayList();

		AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();

		while(it.value() != last) {
			final int v = it.value(); // offset

			final int outV = v / multiplier;
			final int outM = v % multiplier;

			splitOffs[outM].appendValue(outV);

			it.next();
		}

		// last value
		final int v = it.value();
		final int outV = v / multiplier;
		final int outM = v % multiplier;
		splitOffs[outM].appendValue(outV);

		// iterate through all rows.

		AOffset[] offs = new AOffset[multiplier];
		for(int i = 0; i < multiplier; i++)
			offs[i] = OffsetFactory.createOffset(splitOffs[i]);

		// assign columns
		AColGroup[] res = new AColGroup[multiplier];
		for(int i = 0; i < multiplier; i++) {
			final IColIndex ci = i == 0 ? _colIndexes : _colIndexes.shift(i * nColOrg);
			res[i] = create(ci, _numRows / multiplier, _dict, _defaultTuple, offs[i], null);
		}
		return res;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Default: "));
		sb.append(Arrays.toString(_defaultTuple));
		sb.append(String.format("\n%15s", "Indexes: "));
		sb.append(_indexes.toString());
		return sb.toString();
	}
}
