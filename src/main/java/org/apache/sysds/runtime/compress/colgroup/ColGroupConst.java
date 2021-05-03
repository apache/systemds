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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.pre.ArrPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

import edu.emory.mathcs.backport.java.util.Arrays;

public class ColGroupConst extends ColGroupValue {

	private static final long serialVersionUID = 3204391661346504L;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupConst(int numRows) {
		super(numRows);
	}

	public static ColGroupConst genColGroupConst(int numRows, int numCols, double value) {

		int[] colIndices = new int[numCols];
		for(int i = 0; i < numCols; i++)
			colIndices[i] = i;

		double[] values = new double[numCols];
		for(int i = 0; i < numCols; i++)
			values[i] = value;

		ADictionary dict = new Dictionary(values);
		return new ColGroupConst(colIndices, numRows, dict);
	}

	/**
	 * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
	 * 
	 * @param colIndices The Colum indexes for the column group.
	 * @param numRows    The number of rows contained in the group.
	 * @param dict       The dictionary containing one tuple for the entire compression.
	 */
	public ColGroupConst(int[] colIndices, int numRows, ADictionary dict) {
		super(colIndices, numRows, dict, null);
	}

	@Override
	public int[] getCounts(int[] out) {
		out[0] = _numRows;
		return out;
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] out) {
		out[0] = ru - rl;
		return out;
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		double vals = _dict.sumAllRowsToDouble(square, _colIndexes.length)[0];
		for(int rix = rl; rix < ru; rix++)
			c[rix] += vals;
	}

	@Override
	protected void computeColSums(double[] c, boolean square) {
		_dict.colSum(c, getCounts(), _colIndexes, square);
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		double value = _dict.aggregateTuples(builtin, _colIndexes.length)[0];
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], value);
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
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeCONST(getNumCols(), getNumValues(), isLossy());
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(_colIndexes.length * target.getNumRows() + target.getNonZeros());
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		double[] c = target.getDenseBlockValues();
		offT = offT * target.getNumColumns();
		for(int i = rl; i < ru; i++, offT += target.getNumColumns())
			for(int j = 0; j < _colIndexes.length; j++)
				c[offT + _colIndexes[j]] += values[j];
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int ncol = getNumCols();
		double[] values = getValues();
		for(int i = 0; i < _numRows; i++)
			for(int colIx = 0; colIx < ncol; colIx++) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = values[colIx];
				target.quickSetValue(i, col, target.quickGetValue(i, col) + cellVal);
			}

	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colPos) {
		double[] c = target.getDenseBlockValues();
		double v = _dict.getValue(colPos);
		if(v != 0)
			for(int i = 0; i < c.length; i++)
				c[i] += v;

		target.setNonZeros(_numRows);

	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colPos, int rl, int ru) {
		double[] c = target.getDenseBlockValues();
		double v = _dict.getValue(colPos);
		final int length = ru - rl;
		if(v != 0)
			for(int i = 0; i < length; i++)
				c[i] += v;

		target.setNonZeros(_numRows);
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colPos, int rl, int ru) {
		double v = _dict.getValue(colPos);
		final int length = ru - rl;
		if(v != 0)
			for(int i = 0; i < length; i++)
				c[i] += v;

	}

	@Override
	public double get(int r, int c) {
		return _dict.getValue(Arrays.binarySearch(_colIndexes, c));
	}

	// @Override
	// public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
	// double[] vals = preaggValues(1, b, dictVals);
	// for (int i = 0; i < c.length; i++) {
	// c[i] += vals[0];
	// }
	// }

	// @Override
	// public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int
	// rl,
	// int ru) {
	// for (int i = rl * thatNrColumns; i < ru * thatNrColumns; i += thatNrColumns)
	// for (int j = 0; j < outputColumns.length; j++)
	// c[outputColumns[j] + i] += preAggregatedB[j];
	// }

	public double[] preAggregate(double[] a, int row) {
		return new double[] {preAggregateSingle(a, row)};
	}

	public double[] preAggregateSparse(SparseBlock sb, int row) {
		return new double[] {preAggregateSparseSingle(sb, row)};
	}

	public double preAggregateSparseSingle(SparseBlock sb, int row) {
		double v = 0;
		double[] sparseV = sb.values(row);
		for(int i = sb.pos(row); i < sb.pos(row) + sb.size(row); i++) {
			v += sparseV[i];
		}
		return v;
	}

	private double preAggregateSingle(double[] a, int row) {
		double vals = 0;
		for(int off = _numRows * row; off < _numRows * row + _numRows; off++)
			vals += a[off];
		return vals;
	}

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
	// double preAggVals = preAggregateSingle(a, 0);

	// for (int i = 0; i < _colIndexes.length; i++) {
	// c[_colIndexes[i]] += preAggVals * values[i];
	// }
	// }

	@Override
	public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			double preAggVals = preAggregateSingle(a, i);
			int offC = i * numCols;
			for(int j = 0; j < _colIndexes.length; j++) {
				c[offC + _colIndexes[j]] += preAggVals * values[j];
			}
		}
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row) {
		if(!sb.isEmpty(row)) {
			double v = preAggregateSparseSingle(sb, row);
			int offC = row * numCols;
			for(int j = 0; j < _colIndexes.length; j++) {
				c[offC + _colIndexes[j]] += v * values[j];
			}
		}
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupConst(_colIndexes, _numRows, applyScalarOp(op));
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		return new ColGroupConst(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, true, left));
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {

		double[] values = _dict.getValues();
		int base = 0;
		for(int i = 0; i < values.length; i++) {
			base += values[i] == 0 ? 0 : 1;
		}
		for(int i = 0; i < ru - rl; i++) {
			rnnz[i] = base;
		}
	}

	@Override
	public int getIndexStructureHash() {
		throw new NotImplementedException("This function should not be called");
	}

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		return new ArrPreAggregate(lhs.getCounts());
	}

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		throw new DMLCompressionException("Does not make sense to call this");
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret) {
		throw new DMLCompressionException("Does not make sense to call this");
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		throw new DMLCompressionException("Does not make sense to call this");
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		throw new DMLCompressionException("Does not make sense to call this");
	}

	@Override
	protected int containsAllZeroTuple() {
		return -1;
	}

	@Override
	protected boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupEmpty || that instanceof ColGroupConst;
	}
}
