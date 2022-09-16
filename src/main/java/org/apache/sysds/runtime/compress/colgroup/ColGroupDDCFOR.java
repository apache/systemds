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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
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
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 */
public class ColGroupDDCFOR extends AMorphingMMColGroup {
	private static final long serialVersionUID = -5769772089913918987L;

	/** Pointers to row indexes in the dictionary */
	protected AMapToData _data;

	/** Reference values in this column group */
	protected double[] _reference;

	private ColGroupDDCFOR(int[] colIndexes, ADictionary dict, double[] reference, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);
		if(data.getUnique() != dict.getNumberOfValues(colIndexes.length))
			throw new DMLCompressionException("Invalid construction of DDC group " + data.getUnique() + " vs. "
				+ dict.getNumberOfValues(colIndexes.length));
		_data = data;
		_reference = reference;
	}

	public static AColGroup create(int[] colIndexes, ADictionary dict, AMapToData data, int[] cachedCounts,
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
		final int nCol = g.getColIndices().length;
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
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r), colIdx, _colIndexes.length) + _reference[colIdx];
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
		size += 8 * _colIndexes.length;
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
			final ADictionary newDict = _dict.applyScalarOp(op);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else {
			final ADictionary newDict = _dict.applyScalarOpWithReference(op, _reference, newRef);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newRef = ColGroupUtils.unaryOperator(op, _reference);
		final ADictionary newDict = _dict.applyUnaryOpWithReference(op, _reference, newRef);
		return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(v[_colIndexes[i]], _reference[i]);

		if(op.fn instanceof Plus || op.fn instanceof Minus) // only edit reference
			return create(_colIndexes, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final ADictionary newDict = _dict.binOpLeftWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(_reference[i], v[_colIndexes[i]]);

		if(op.fn instanceof Plus || op.fn instanceof Minus)// only edit reference
			return create(_colIndexes, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final ADictionary newDict = _dict.binOpRightWithReference(op, v, _colIndexes, _reference, newRef);
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
		int[] cols = AColGroup.readCols(in);
		ADictionary dict = DictionaryFactory.read(in);
		AMapToData data = MapToFactory.readIn(in);
		double[] ref = ColGroupIO.readDoubleArray(cols.length, in);
		return new ColGroupDDCFOR(cols, dict, ref, data, null);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _data.getExactSizeOnDisk();
		ret += 8 * _colIndexes.length; // reference values.
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
		final ADictionary newDict = _dict.replaceWithReference(pattern, replace, _reference);
		boolean patternInReference = false;
		for(double d : _reference)
			if(pattern == d) {
				patternInReference = true;
				break;
			}
		if(patternInReference) {
			double[] nRef = new double[_reference.length];
			for(int i = 0; i < _reference.length; i++)
				if(pattern == _reference[i])
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
		for(int i = 0; i < _colIndexes.length; i++)
			c[_colIndexes[i]] += _reference[i] * nRows;
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
	protected AColGroup sliceSingleColumn(int idx) {
		ColGroupDDCFOR ret = (ColGroupDDCFOR) super.sliceSingleColumn(idx);
		// select values from double array.
		ret._reference = new double[1];
		ret._reference[0] = _reference[idx];
		return ret;
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		ColGroupDDCFOR ret = (ColGroupDDCFOR) super.sliceMultiColumns(idStart, idEnd, outputCols);
		final int len = idEnd - idStart;
		ret._reference = new double[len];
		for(int i = 0, ii = idStart; i < len; i++, ii++)
			ret._reference[i] = _reference[ii];

		return ret;
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
		return (long) _colIndexes.length * nRows;
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += _reference[i];
		return ColGroupDDC.create(_colIndexes, _dict, _data, getCounts());
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		final int def = (int) _reference[0];
		ADictionary d = _dict.rexpandColsWithReference(max, ignore, cast, def);

		if(d == null) {
			if(def <= 0 || def > max)
				return ColGroupEmpty.create(max);
			else {
				double[] retDef = new double[max];
				retDef[((int) def) - 1] = 1;
				return ColGroupConst.create(retDef);
			}
		}
		else {
			int[] outCols = Util.genColsIndices(max);
			if(def <= 0) {
				if(ignore)
					return ColGroupDDC.create(outCols, d, _data, getCachedCounts());
				else
					throw new DMLRuntimeException("Invalid content of zero in rexpand");
			}
			else if(def > max)
				return ColGroupDDC.create(outCols, d, _data, getCachedCounts());
			else {
				double[] retDef = new double[max];
				retDef[((int) def) - 1] = 1;
				return ColGroupDDCFOR.create(outCols, d, _data, getCachedCounts(), retDef);
			}
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
	protected AColGroup allocateRightMultiplicationCommon(double[] common, int[] colIndexes, ADictionary preAgg) {
		return create(colIndexes, preAgg, _data, getCachedCounts(), common);
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
