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

import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

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
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
		double vals = _dict.sumAllRowsToDouble(square, _colIndexes.length)[0];
		for(int rix = rl; rix < ru; rix++)
			c[rix] += vals;
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
	protected void decompressToBlockUnSafeDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values) {
		double[] c = target.getDenseBlockValues();
		offT = offT * target.getNumColumns();
		for(int i = rl; i < ru; i++, offT += target.getNumColumns())
			for(int j = 0; j < _colIndexes.length; j++)
				c[offT + _colIndexes[j]] += values[j];
	}

	@Override
	protected void decompressToBlockUnSafeSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock values) {
		throw new NotImplementedException();
	}

	@Override
	public double get(int r, int c) {
		return _dict.getValue(Arrays.binarySearch(_colIndexes, c));
	}

	@Override
	protected void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru) {
		if(m.isInSparseFormat())
			preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
		else
			preAggregateDense(m, preAgg, rl, ru);
	}

	private void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		final int numVals = getNumValues();
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
			for(int rc = 0, offLeft = rowLeft * _numRows; rc < _numRows; rc++, offLeft++) {
				preAV[offOut] += mV[offLeft];
			}
		}
	}

	private void preAggregateSparse(SparseBlock sb, MatrixBlock preAgg, int rl, int ru) {
		final double[] preAV = preAgg.getDenseBlockValues();
		final int numVals = getNumValues();
		for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += numVals) {
			if(sb.isEmpty(rowLeft))
				continue;
			final int apos = sb.pos(rowLeft);
			final int alen = sb.size(rowLeft) + apos;
			final double[] avals = sb.values(rowLeft);
			for(int j = apos; j < alen; j++) {
				preAV[offOut] += avals[j];
			}
		}
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		return new ColGroupConst(_colIndexes, _numRows, applyScalarOp(op));
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		return new ColGroupConst(_colIndexes, _numRows, applyBinaryRowOp(op, v, true, left));
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
	protected boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupEmpty || that instanceof ColGroupConst;
	}
}
