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
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class ColGroupConst extends AColGroupCompressed {

	private static final long serialVersionUID = -7387793538322386611L;

	protected ADictionary _dict;

	/** Constructor for serialization */
	protected ColGroupConst() {
		super();
	}

	/**
	 * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
	 * 
	 * @param colIndices The Colum indexes for the column group.
	 * @param dict       The dictionary containing one tuple for the entire compression.
	 */
	private ColGroupConst(int[] colIndices, ADictionary dict) {
		super(colIndices);
		this._dict = dict;
	}

	/**
	 * Create constructor for a ColGroup Const this constructor ensures that if the dictionary input is empty an Empty
	 * column group is constructed.
	 * 
	 * @param colIndices The column indexes in the column group
	 * @param dict       The dictionary to use
	 * @return A Colgroup either const or empty.
	 */
	protected static AColGroup create(int[] colIndices, ADictionary dict) {
		if(dict == null)
			return new ColGroupEmpty(colIndices);
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
		final int[] colIndices = Util.genColsIndices(values.length);
		return create(colIndices, values);
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
	public static AColGroup create(int[] cols, double value) {
		final int numCols = cols.length;
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
	public static AColGroup create(int[] cols, double[] values) {
		if(cols.length != values.length)
			throw new DMLCompressionException("Invalid size of values compared to columns");
		ADictionary dict = new Dictionary(values);
		return ColGroupConst.create(cols, dict);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns.
	 * @param dict    The dictionary to contain int the Constant group.
	 * @return A Constant column group.
	 */
	public static AColGroup create(int numCols, ADictionary dict) {
		if(numCols != dict.getValues().length)
			throw new DMLCompressionException(
				"Invalid construction of const column group with different number of columns in arguments");
		final int[] colIndices = Util.genColsIndices(numCols);
		return ColGroupConst.create(colIndices, dict);
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
		final int[] colIndices = Util.genColsIndices(numCols);

		if(value == 0)
			return new ColGroupEmpty(colIndices);
		return ColGroupConst.create(colIndices, value);
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

	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		if(db.isContiguous() && _colIndexes.length == db.getDim(1) && offC == 0)
			decompressToDenseBlockAllColumnsContiguous(db, rl, ru, offR, offC);
		else
			decompressToDenseBlockGeneric(db, rl, ru, offR, offC);
	}

	private void decompressToDenseBlockAllColumnsContiguous(DenseBlock db, int rl, int ru, int offR, int offC) {

		final double[] c = db.values(0);
		final int nCol = _colIndexes.length;
		final double[] values = _dict.getValues();
		for(int r = rl; r < ru; r++) {
			final int offStart = (offR + r) * nCol;
			for(int vOff = 0, off = offStart; vOff < nCol; vOff++, off++)
				c[off] += values[vOff];
		}
	}

	private void decompressToDenseBlockGeneric(DenseBlock db, int rl, int ru, int offR, int offC) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < _colIndexes.length; j++)
				c[off + _colIndexes[j]] += _dict.getValue(j);
		}
	}

	@Override
	public void decompressToSparseBlock(SparseBlock ret, int rl, int ru, int offR, int offC) {
		final int nCol = _colIndexes.length;
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes[j] + offC, _dict.getValue(j));
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
	public void addToCommon(double[] constV) {
		final double[] values = _dict.getValues();
		for(int i = 0; i < _colIndexes.length; i++)
			constV[_colIndexes[i]] += values[i];
	}

	public double[] getValues() {
		return _dict.getValues();
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
		c[0] += _dict.sum(new int[] {nRows}, _colIndexes.length);
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		_dict.colSum(c, new int[] {nRows}, _colIndexes);
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSq(new int[] {nRows}, _colIndexes.length);
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

	private synchronized MatrixBlock forceValuesToMatrixBlock() {
		_dict = _dict.getMBDict(_colIndexes.length);
		MatrixBlock ret = ((MatrixBlockDictionary) _dict).getMatrixBlock();
		return ret;
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		if(right.isEmpty())
			return null;
		final int rr = right.getNumRows();
		final int cr = right.getNumColumns();
		if(_colIndexes.length == rr) {
			final MatrixBlock left = forceValuesToMatrixBlock();
			if(left == null)
				return null;
			final MatrixBlock ret = new MatrixBlock(1, cr, false);
			LibMatrixMult.matrixMult(left, right, ret);
			if(ret.isEmpty())
				return null;
			final ADictionary d = new MatrixBlockDictionary(ret, cr);
			return create(cr, d);
		}
		else {
			throw new NotImplementedException();
		}
	}

	@Override
	public void tsmm(double[] result, int numColumns, int nRows) {
		tsmm(result, numColumns, new int[] {nRows}, _dict, _colIndexes);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called");
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called");
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		int[] colIndexes = new int[] {0};
		double v = _dict.getValue(idx);
		if(v == 0)
			return new ColGroupEmpty(colIndexes);
		else {
			ADictionary retD = new Dictionary(new double[] {_dict.getValue(idx)});
			return create(colIndexes, retD);
		}
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		ADictionary retD = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.length);
		return create(outputCols, retD);
	}

	@Override
	public AColGroup copy() {
		return create(_colIndexes, _dict.clone());
	}

	@Override
	public boolean containsValue(double pattern) {
		return _dict.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		return _dict.getNumberNonZeros(new int[] {nRows}, _colIndexes.length);
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		return create(_colIndexes, replaced);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_dict = DictionaryFactory.read(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_dict.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _dict.getExactSizeOnDisk();
		return ret;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		_dict.product(c, new int[] {nRows}, _colIndexes.length);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDouble(_colIndexes.length);
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSq(_colIndexes.length);
	}

	@Override
	protected double[] preAggProductRows() {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRows(builtin, _colIndexes.length);
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _dict.getInMemorySize();
		size += 8; // dict reference
		return size;
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		CM_COV_Object ret = new CM_COV_Object();
		op.fn.execute(ret, _dict.getValue(0), nRows);
		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		ADictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.length);
		if(d == null)
			return ColGroupEmpty.create(max);
		else
			return create(max, d);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nCols = getNumCols();
		return e.getCost(nRows, 1, nCols, 1, 1.0);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Values: " + _dict.getClass().getSimpleName()));
		sb.append(_dict.getString(_colIndexes.length));
		return sb.toString();
	}
}
