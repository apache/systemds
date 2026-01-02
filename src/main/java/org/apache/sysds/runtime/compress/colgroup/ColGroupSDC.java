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
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Column group that sparsely encodes the dictionary values. The idea is that all values is encoded with indexes except
 * the most common one. the most common one can be inferred by not being included in the indexes.
 * 
 * This column group is handy in cases where sparse unsafe operations is executed on very sparse columns. Then the zeros
 * would be materialized in the group without any overhead.
 */
public class ColGroupSDC extends ASDC implements IMapToDataGroup {
	private static final long serialVersionUID = 769993538831949086L;

	/** Pointers to row indexes in the dictionary. */
	protected final AMapToData _data;
	/** The default value stored in this column group */
	protected final double[] _defaultTuple;

	protected ColGroupSDC(IColIndex colIndices, int numRows, IDictionary dict, double[] defaultTuple, AOffset offsets,
		AMapToData data, int[] cachedCounts) {
		super(colIndices, numRows, dict, offsets, cachedCounts);
		_data = data;
		_defaultTuple = defaultTuple;
		if(CompressedMatrixBlock.debug) {

			if(data.getUnique() != dict.getNumberOfValues(colIndices.size())) {
				if(data.getUnique() != data.getMax())
					throw new DMLCompressionException(
						"Invalid unique count compared to actual: " + data.getUnique() + " " + data.getMax());
				throw new DMLCompressionException("Invalid construction of SDC group: number uniques: " + data.getUnique()
					+ " vs." + dict.getNumberOfValues(colIndices.size()));
			}
			if(_indexes.getSize() == numRows)
				LOG.warn("Invalid SDC group that contains index with size == numRows");
			if(defaultTuple.length != colIndices.size())
				throw new DMLCompressionException("Invalid construction of SDC group");

			_data.verify();
			_indexes.verify(_data.size());
		}

	}

	public static AColGroup create(IColIndex colIndices, int numRows, IDictionary dict, double[] defaultTuple,
		AOffset offsets, AMapToData data, int[] cachedCounts) {
		final boolean allZero = ColGroupUtils.allZero(defaultTuple);
		if(dict == null && allZero)
			return new ColGroupEmpty(colIndices);
		else if(dict == null || dict.getNumberOfValues(colIndices.size()) == 1)
			return ColGroupSDCSingle.create(colIndices, numRows, dict, defaultTuple, offsets, null);
		else if(allZero)
			return ColGroupSDCZeros.create(colIndices, numRows, dict, offsets, data, cachedCounts);
		else if(data.getUnique() == 1) {
			MatrixBlock mb = dict.getMBDict(colIndices.size()).getMatrixBlock().slice(0, 0);
			return ColGroupSDCSingle.create(colIndices, numRows, MatrixBlockDictionary.create(mb), defaultTuple, offsets,
				null);
		}
		else
			return new ColGroupSDC(colIndices, numRows, dict, defaultTuple, offsets, data, cachedCounts);
	}

	public AColGroup sparsifyFOR() {
		return ColGroupSDCFOR.sparsifyFOR(this);
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.SDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.SDC;
	}

	@Override
	public double[] getDefaultTuple() {
		return _defaultTuple;
	}

	@Override
	public AMapToData getMapToData() {
		return _data;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final AIterator it = _indexes.getIterator(r);
		if(it == null || it.value() != r)
			return _defaultTuple[colIdx];
		else
			return _dict.getValue(_data.getIndex(it.getDataIndex()), colIdx, _colIndexes.size());

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
		computeRowSums(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	protected static final void computeRowSums(double[] c, int rl, int ru, double[] preAgg, AMapToData data,
		AOffset indexes, int nRows) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		final double def = preAgg[preAgg.length - 1];
		if(it != null) {

			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxId = data.size() - 1;

				while(true) {
					if(it.value() == r) {
						c[r] += preAgg[data.getIndex(it.getDataIndex())];
						if(it.getDataIndex() < maxId)
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
						c[r++] += preAgg[data.getIndex(it.getDataIndex())];
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
		computeRowMxx(c, builtin, rl, ru, preAgg, _data, _indexes, _numRows, preAgg[preAgg.length - 1]);
	}

	protected static final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg,
		AMapToData data, AOffset indexes, int nRows, double def) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		if(it != null) {
			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxId = data.size() - 1;
				while(true) {
					if(it.value() == r) {
						c[r] = builtin.execute(c[r], preAgg[data.getIndex(it.getDataIndex())]);
						if(it.getDataIndex() < maxId)
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
						c[r] = builtin.execute(c[r], preAgg[data.getIndex(it.getDataIndex())]);
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
	protected void computeSum(double[] c, int nRows) {
		super.computeSum(c, nRows);
		int count = _numRows - _data.size();
		for(int x = 0; x < _defaultTuple.length; x++)
			c[0] += _defaultTuple[x] * count;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		super.computeColSums(c, nRows);
		int count = _numRows - _data.size();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] += _defaultTuple[x] * count;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		super.computeSumSq(c, nRows);
		int count = _numRows - _data.size();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[0] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		super.computeColSumsSq(c, nRows);
		int count = _numRows - _data.size();
		for(int x = 0; x < _colIndexes.size(); x++)
			c[_colIndexes.get(x)] += _defaultTuple[x] * _defaultTuple[x] * count;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		final int count = _numRows - _data.size();
		_dict.productWithDefault(c, getCounts(), _defaultTuple, count);
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		super.computeColProduct(c, nRows);
		final int count = _numRows - _data.size();
		for(int x = 0; x < _colIndexes.size(); x++) {
			double v = c[_colIndexes.get(x)];
			c[_colIndexes.get(x)] = v != 0 ? v * Math.pow(_defaultTuple[x], count) : 0;
		}
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		computeRowProduct(c, rl, ru, preAgg, _data, _indexes, _numRows);
	}

	protected static final void computeRowProduct(double[] c, int rl, int ru, double[] preAgg, AMapToData data,
		AOffset indexes, int nRows) {
		int r = rl;
		final AIterator it = indexes.getIterator(rl);
		final double def = preAgg[preAgg.length - 1];
		if(it != null) {

			if(it.value() > ru)
				indexes.cacheIterator(it, ru);
			else if(ru > indexes.getOffsetToLast()) {
				final int maxId = data.size() - 1;
				while(true) {
					if(it.value() == r) {
						c[r] *= preAgg[data.getIndex(it.getDataIndex())];
						if(it.getDataIndex() < maxId)
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
						c[r] *= preAgg[data.getIndex(it.getDataIndex())];
						it.next();
					}
					else
						c[r] *= def;
					r++;
				}
				indexes.cacheIterator(it, ru);
			}
		}

		while(r < ru) {
			c[r] *= def;
			r++;
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		return _data.getCounts(counts);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		long c = super.getNumberNonZeros(nRows);
		int count = _numRows - _data.size();
		for(int x = 0; x < _colIndexes.size(); x++)
			c += _defaultTuple[x] != 0 ? count : 0;
		return c;
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
	public AColGroup scalarOperation(ScalarOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.executeScalar(_defaultTuple[i]);
		final IDictionary nDict = _dict.applyScalarOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i]);
		final IDictionary nDict = _dict.applyUnaryOp(op);
		return create(_colIndexes, _numRows, nDict, newDefaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(v[_colIndexes.get(i)], _defaultTuple[i]);
		final IDictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
		return create(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newDefaultTuple = new double[_defaultTuple.length];
		for(int i = 0; i < _defaultTuple.length; i++)
			newDefaultTuple[i] = op.fn.execute(_defaultTuple[i], v[_colIndexes.get(i)]);
		final IDictionary newDict = _dict.binOpRight(op, v, _colIndexes);
		return create(_colIndexes, _numRows, newDict, newDefaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_indexes.write(out);
		_data.write(out);
		for(double d : _defaultTuple)
			out.writeDouble(d);
	}

	public static ColGroupSDC read(DataInput in, int nRows) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AOffset indexes = OffsetFactory.readIn(in);
		AMapToData data = MapToFactory.readIn(in);
		double[] defaultTuple = ColGroupIO.readDoubleArray(cols.size(), in);
		return new ColGroupSDC(cols, nRows, dict, defaultTuple, indexes, data, null);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
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

		return create(_colIndexes, _numRows, replaced, newDefaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.size(); i++)
			constV[_colIndexes.get(i)] += _defaultTuple[i];

		IDictionary subtractedDict = _dict.subtractTuple(_defaultTuple);
		return ColGroupSDCZeros.create(_colIndexes, _numRows, subtractedDict, _indexes, _data, getCounts());
	}

	public AColGroup subtractDefaultTuple() {
		IDictionary subtractedDict = _dict.subtractTuple(_defaultTuple);
		return ColGroupSDCZeros.create(_colIndexes, _numRows, subtractedDict, _indexes, _data, getCounts());
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		return _dict.centralMomentWithDefault(op.fn, getCounts(), _defaultTuple[0], nRows);
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		IDictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.size());
		return rexpandCols(max, ignore, cast, nRows, d, _indexes, _data, getCachedCounts(), (int) _defaultTuple[0],
			_dict.getNumberOfValues(1));
	}

	protected static AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows, IDictionary d,
		AOffset indexes, AMapToData data, int[] counts, int def, int nVal) {

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
				retDef[def - 1] = 1;
				return ColGroupSDCSingle.create(ColIndexFactory.create(max), nRows, Dictionary.create(new double[max]),
					retDef, indexes, null);
			}
		}
		else {
			final IColIndex outCols = ColIndexFactory.create(d.getNumberOfColumns(nVal));
			if(def <= 0) {
				if(ignore)
					return ColGroupSDCZeros.create(outCols, nRows, d, indexes, data, counts);
				else
					throw new DMLRuntimeException("Invalid content of zero in rexpand");
			}
			else if(def > max)
				return ColGroupSDCZeros.create(outCols, nRows, d, indexes, data, counts);
			else {
				double[] retDef = new double[max];
				retDef[def - 1] = 1;
				return ColGroupSDC.create(outCols, nRows, d, retDef, indexes, data, counts);
			}
		}
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		final int nRowsScanned = _data.size();
		return e.getCost(nRows, nRowsScanned, nCols, nVals, _dict.getSparsity());
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		IDictionary retDict = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.size());
		final double[] newDef = new double[idEnd - idStart];
		for(int i = idStart, j = 0; i < idEnd; i++, j++)
			newDef[j] = _defaultTuple[i];
		return create(outputCols, _numRows, retDict, newDef, _indexes, _data, getCounts());
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		final IColIndex retIndexes = ColIndexFactory.create(1);
		if(_colIndexes.size() == 1) // early abort, only single column already.
			return create(retIndexes, _numRows, _dict, _defaultTuple, _indexes, _data, getCounts());
		final double[] newDef = new double[] {_defaultTuple[idx]};
		final IDictionary retDict = _dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.size());
		return create(retIndexes, _numRows, retDict, newDef, _indexes, _data, getCounts());
	}

	@Override
	public boolean containsValue(double pattern) {
		if(_dict.containsValue(pattern))
			return true;
		for(double v : _defaultTuple)
			if(v == pattern)
				return true;
		return false;
	}

	@Override
	public double[] getCommon() {
		return _defaultTuple;
	}

	@Override
	protected AColGroup allocateRightMultiplicationCommon(double[] common, IColIndex colIndexes, IDictionary preAgg) {
		return create(colIndexes, _numRows, preAgg, common, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {

		if(ru > _numRows)
			throw new DMLRuntimeException("Invalid row range");
		OffsetSliceInfo off = _indexes.slice(rl, ru);
		if(off.lIndex == -1)
			return ColGroupConst.create(_colIndexes, Dictionary.create(_defaultTuple));
		AMapToData newData = _data.slice(off.lIndex, off.uIndex);
		return create(_colIndexes, ru - rl, _dict, _defaultTuple, off.offsetSlice, newData, null);
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, _numRows, newDictionary, _defaultTuple, _indexes, _data, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupSDC && g.getColIndices().equals(_colIndexes)) {
			final ColGroupSDC gSDC = (ColGroupSDC) g;
			if(Arrays.equals(_defaultTuple, gSDC._defaultTuple) && gSDC._dict.equals(_dict)) {
				final AMapToData nd = _data.append(gSDC._data);
				final AOffset ofd = _indexes.append(gSDC._indexes, getNumRows());
				return create(_colIndexes, _numRows + gSDC._numRows, _dict, _defaultTuple, ofd, nd, null);
			}
		}
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
		AMapToData nd = _data.appendN(Arrays.copyOf(g, g.length, IMapToDataGroup[].class));
		AOffset no = _indexes.appendN(Arrays.copyOf(g, g.length, AOffsetsGroup[].class), getNumRows());

		return create(_colIndexes, rlen, _dict, _defaultTuple, no, nd, null);
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
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return ColGroupSDC.create(newColIndex, getNumRows(), _dict.reorder(reordering),
			ColGroupUtils.reorderDefault(_defaultTuple, reordering), _indexes, _data, getCachedCounts());
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that instanceof ColGroupSDCZeros) {
			ColGroupSDCZeros th = (ColGroupSDCZeros) that;
			return th._indexes == _indexes && th._data == _data;
		}
		else if(that instanceof ColGroupSDC) {
			ColGroupSDC th = (ColGroupSDC) that;
			return th._indexes == _indexes && th._data == _data;
		}
		else
			return false;
	}

	@Override
	public int getNumberOffsets() {
		return _data.size();
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sr = ret.getSparseBlock();
		final int nCol = _colIndexes.size();
		final AIterator it = _indexes.getIterator();
		final int last = _indexes.getOffsetToLast();

		int c = 0;

		int of = it.value();
		while(of < last && c < points.length) {
			// final int of = it.value();
			if(points[c].o < of) {
				putDefault(points[c].r, sr, nCol);
				c++;
			}
			else {
				while(c < points.length && points[c].o == of) {
					_dict.putSparse(sr, _data.getIndex(it.getDataIndex()), points[c].r, nCol, _colIndexes);
					c++;
				}
				of = it.next();
			}
			// c++;
		}

		for(; c < points.length && points[c].o < last; c++) {
			putDefault(points[c].r, sr, nCol);
		}

		while(of == last && c < points.length && points[c].o == of) {
			_dict.putSparse(sr, _data.getIndex(it.getDataIndex()), points[c].r, nCol, _colIndexes);
			c++;
		}

		// set default in tail.
		for(; c < points.length; c++) {
			putDefault(points[c].r, sr, nCol);
		}

	}

	private void putDefault(final int r, final SparseBlock sr, final int nCol) {
		if(sr.isAllocated(r))
			for(int i = 0; i < nCol; i++)
				sr.add(r, _colIndexes.get(i), _defaultTuple[i]);
		else {
			sr.allocate(r, _colIndexes.size());
			for(int i = 0; i < nCol; i++)
				sr.append(r, _colIndexes.get(i), _defaultTuple[i]);
		}

	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup morph(CompressionType ct, int nRow) {
		if(ct == getCompType())
			return this;
		else if(ct == CompressionType.DDC) {

			AMapToData nMap = MapToFactory.create(nRow, _data.getUnique() + 1);
			IDictionary nDict = _dict.append(_defaultTuple);

			final AIterator it = _indexes.getIterator();
			final int last = _indexes.getOffsetToLast();
			int r = 0;
			int def = _data.getUnique();
			while(it.value() < last) {
				final int iv = it.value();
				while(r < iv) {
					nMap.set(r++, def);
				}
				nMap.set(r++, _data.getIndex(it.getDataIndex()));
				it.next();
			}
			nMap.set(r++, _data.getIndex(it.getDataIndex()));
			while(r < nRow) {
				nMap.set(r++, def);
			}

			return ColGroupDDC.create(_colIndexes, nDict, nMap, null);
		}

		else {
			return super.morph(ct, nRow);
		}
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {

		final IDictionary combined = combineDictionaries(nCol, right);
		final IColIndex combinedColIndex = combineColIndexes(nCol, right);
		final double[] combinedDefaultTuple = IContainDefaultTuple.combineDefaultTuples(_defaultTuple, right);

		// return new ColGroupDDC(combinedColIndex, combined, _data, getCachedCounts());
		return new ColGroupSDC(combinedColIndex, this.getNumRows(), combined, combinedDefaultTuple, _indexes, _data,
			getCachedCounts());
	}

	@Override
	public AColGroupCompressed combineWithSameIndex(int nRow, int nCol, AColGroup right) {
		if(right instanceof ColGroupSDCZeros) {
			ColGroupSDCZeros rightSDC = ((ColGroupSDCZeros) right);
			IDictionary b = rightSDC.getDictionary();
			IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
			IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
			double[] combinedDefaultTuple = new double[_defaultTuple.length + right.getNumCols()];
			System.arraycopy(_defaultTuple, 0, combinedDefaultTuple, 0, _defaultTuple.length);
			// System.arraycopy(rightSDC._defaultTuple, 0, combinedDefaultTuple, _defaultTuple.length,
			// rightSDC._defaultTuple.length);

			return new ColGroupSDC(combinedColIndex, this.getNumRows(), combined, combinedDefaultTuple, _indexes, _data,
				getCachedCounts());
		}
		else {
			ColGroupSDC rightSDC = ((ColGroupSDC) right);
			IDictionary b = rightSDC.getDictionary();
			IDictionary combined = DictionaryFactory.cBindDictionaries(_dict, b, this.getNumCols(), right.getNumCols());
			IColIndex combinedColIndex = _colIndexes.combine(right.getColIndices().shift(nCol));
			double[] combinedDefaultTuple = new double[_defaultTuple.length + rightSDC._defaultTuple.length];
			System.arraycopy(_defaultTuple, 0, combinedDefaultTuple, 0, _defaultTuple.length);
			System.arraycopy(rightSDC._defaultTuple, 0, combinedDefaultTuple, _defaultTuple.length,
				rightSDC._defaultTuple.length);

			return new ColGroupSDC(combinedColIndex, this.getNumRows(), combined, combinedDefaultTuple, _indexes, _data,
				getCachedCounts());
		}
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
			res[i] = create(ci, _numRows / multiplier, _dict, _defaultTuple, offs[i], maps[i], null);
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
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(_data.toString());
		return sb.toString();
	}
}
