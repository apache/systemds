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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.utils.Util;
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
public class ColGroupDDCFOR extends AMorphingMMColGroup {
	private static final long serialVersionUID = -5769772089913918987L;

	/** Pointers to row indexes in the dictionary */
	protected AMapToData _data;

	/** Reference values in this column group */
	protected double[] _reference;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows number of rows
	 */
	protected ColGroupDDCFOR(int numRows) {
		super(numRows);
	}

	private ColGroupDDCFOR(int[] colIndexes, int numRows, ADictionary dict, double[] reference, AMapToData data,
		int[] cachedCounts) {
		super(colIndexes, numRows, dict, cachedCounts);
		if(data.getUnique() != dict.getNumberOfValues(colIndexes.length))
			throw new DMLCompressionException("Invalid construction of DDC group " + data.getUnique() + " vs. "
				+ dict.getNumberOfValues(colIndexes.length));
		_zeros = false;
		_data = data;
		_reference = reference;
	}

	protected static AColGroup create(int[] colIndexes, int numRows, ADictionary dict, AMapToData data,
		int[] cachedCounts, double[] reference) {
		final boolean allZero = FORUtil.allZero(reference);
		if(dict == null && allZero)
			return new ColGroupEmpty(colIndexes);
		else if(dict == null)
			return ColGroupConst.create(colIndexes, reference);
		else if(allZero)
			return ColGroupDDC.create(colIndexes, numRows, dict, data, cachedCounts);
		else
			return new ColGroupDDCFOR(colIndexes, numRows, dict, reference, data, cachedCounts);
	}

	public CompressionType getCompType() {
		return CompressionType.DDCFOR;
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _dict.getValue(_data.getIndex(r) * _colIndexes.length + colIdx) + _reference[colIdx];
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
			return create(_colIndexes, _numRows, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			final ADictionary newDict = _dict.applyScalarOp(op);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
		else {
			final ADictionary newDict = _dict.applyScalarOpWithReference(op, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double[] newRef = FORUtil.unaryOperator(op, _reference);
		final ADictionary newDict = _dict.applyUnaryOpWithReference(op, _reference, newRef);
		return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(v[_colIndexes[i]], _reference[i]);

		if(op.fn instanceof Plus || op.fn instanceof Minus) // only edit reference
			return create(_colIndexes, _numRows, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final ADictionary newDict = _dict.binOpLeft(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final ADictionary newDict = _dict.binOpLeftWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		final double[] newRef = new double[_reference.length];
		for(int i = 0; i < _reference.length; i++)
			newRef[i] = op.fn.execute(_reference[i], v[_colIndexes[i]]);

		if(op.fn instanceof Plus || op.fn instanceof Minus)// only edit reference
			return create(_colIndexes, _numRows, _dict, _data, getCachedCounts(), newRef);
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// possible to simply process on dict and keep reference
			final ADictionary newDict = _dict.binOpRight(op, v, _colIndexes);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
		else { // have to apply reference while processing
			final ADictionary newDict = _dict.binOpRightWithReference(op, v, _colIndexes, _reference, newRef);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), newRef);
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_data.write(out);
		for(double d : _reference)
			out.writeDouble(d);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_data = MapToFactory.readIn(in);
		_reference = new double[_colIndexes.length];
		for(int i = 0; i < _colIndexes.length; i++)
			_reference[i] = in.readDouble();
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
		boolean patternInReference = false;
		for(double d : _reference)
			if(pattern == d) {
				patternInReference = true;
				break;
			}

		if(patternInReference) {
			throw new NotImplementedException("Not Implemented replace where a value in reference should be replaced");
			// _dict.replace(pattern, replace, _reference, _newReplace);
		}
		else {
			final ADictionary newDict = _dict.replaceWithReference(pattern, replace, _reference);
			return create(_colIndexes, _numRows, newDict, _data, getCachedCounts(), _reference);
		}
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
		// trick,use normal sum
		super.computeSum(c, nRows);
		// and add all sum of reference multiplied with nrows.
		final double refSum = FORUtil.refSum(_reference);
		c[0] += refSum * nRows;
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		// trick, use the normal sum
		super.computeColSums(c, nRows);
		// and add reference multiplied with number of rows.
		for(int i = 0; i < _colIndexes.length; i++)
			c[_colIndexes[i]] += _reference[i] * nRows;
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		// square sum the dictionary.
		c[0] += _dict.sumSqWithReference(getCounts(), _reference);
		final double refSum = FORUtil.refSumSq(_reference);
		;
		// Square sum of the reference values only for the rows that is not represented in the Offsets.
		c[0] += refSum * (_numRows - _data.size());
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict = _dict.getMBDict(_colIndexes.length);
		// square sum the dictionary
		_dict.colSumSqWithReference(c, getCounts(), _colIndexes, _reference);
		// Square sum of the reference values only for the rows that is not represented in the Offsets.
		for(int i = 0; i < _colIndexes.length; i++) // correct for the reference sum.
			c[_colIndexes[i]] += _reference[i] * _reference[i] * (_numRows - _data.size());
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
		throw new NotImplementedException();
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
		throw new NotImplementedException("Not Implemented PFOR");
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		throw new NotImplementedException("Not Implemented PFOR");
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
		if(pattern == 0 && _zeros)
			return true;
		else if(Double.isNaN(pattern) || Double.isInfinite(pattern))
			return FORUtil.containsInfOrNan(pattern, _reference) || _dict.containsValue(pattern);
		else
			return _dict.containsValueWithReference(pattern, _reference);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		long nnz = 0;
		int refCount= 0;
		for(int i = 0; i < _reference.length; i ++)
			if(_reference[i] != 0)
				refCount ++;

		if (refCount == _colIndexes.length)
			return (long)_colIndexes.length * nRows;
		else{
			nnz += _dict.getNumberNonZerosWithReference(getCounts(), _reference, nRows);
			nnz += refCount * nRows;
		}

		return Math.min((long)_colIndexes.length * nRows, nnz);
	}

	@Override
	public AColGroup extractCommon(double[] constV) {
		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += _reference[i];
		return ColGroupDDC.create(_colIndexes, _numRows, _dict, _data, getCounts());
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		final double def = _reference[0];
		ADictionary d = _dict.rexpandColsWithReference(max, ignore, cast, def);

		// return ColGroupDDC.rexpandCols(max, ignore, cast, nRows, d, _data, getCachedCounts(), _reference[0]);
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
					return ColGroupDDC.create(outCols, nRows, d, _data, getCachedCounts());
				else
					throw new DMLRuntimeException("Invalid content of zero in rexpand");
			}
			else if(def > max)
				return ColGroupDDC.create(outCols, nRows, d, _data, getCachedCounts());
			else {
				double[] retDef = new double[max];
				retDef[((int) def) - 1] = 1;
				return create(outCols, nRows, d, _data, getCachedCounts(), retDef);
			}
		}
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		// should be guaranteed to be one column therefore only one reference value.
		CM_COV_Object ret = _dict.centralMomentWithReference(op.fn, getCounts(), _reference[0], nRows);
		int count = _numRows - _data.size();
		op.fn.execute(ret, _reference[0], count);
		return ret;
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
