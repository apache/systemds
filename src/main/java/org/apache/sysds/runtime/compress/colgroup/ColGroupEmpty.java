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

import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class ColGroupEmpty extends ColGroupCompressed {

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupEmpty(int numRows) {
		super(numRows);
	}

	/**
	 * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
	 * 
	 * @param colIndices The Colum indexes for the column group.
	 * @param numRows    The number of rows contained in the group.
	 */
	public ColGroupEmpty(int[] colIndices, int numRows) {
		super(colIndices, numRows);
	}

	public static ColGroupEmpty generate(int nCol, int nRow) {
		int[] cols = new int[nCol];
		for(int i = 0; i < nCol; i++) {
			cols[i] = i;
		}
		return new ColGroupEmpty(cols, nRow);
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {
		// do nothing
	}

	@Override
	protected void computeColSums(double[] c, boolean square) {
		// do nothing
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], 0);

	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.CONST;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.EMPTY;
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT) {
		// do nothing.
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT) {
		// do nothing.
	}

	@Override
	public double get(int r, int c) {
		return 0;
	}

	@Override
	public void leftMultByMatrix(MatrixBlock a, MatrixBlock c, int rl, int ru) {
		// do nothing.
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		if(val0 == 0)
			return this;
		return new ColGroupConst(_colIndexes, _numRows, new Dictionary(new double[_colIndexes.length]).apply(op));
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		if(sparseSafe)
			return this;
		return new ColGroupConst(_colIndexes, _numRows,
			new Dictionary(new double[_colIndexes.length]).applyBinaryRowOp(op, v, true, _colIndexes, left));
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		// do nothing.
	}

	@Override
	public int getNumValues() {
		return 0;
	}

	@Override
	public double[] getValues() {
		return null;
	}

	@Override
	public void addMinMax(double[] ret) {
		// do nothing
	}

	@Override
	public boolean isLossy() {
		return false;
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return builtin.execute(c, 0);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		for(int col : _colIndexes)
			c[col] = builtin.execute(c[col], 0);
	}

	@Override
	protected void computeSum(double[] c, boolean square) {
		// do nothing
	}

	@Override
	protected boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupEmpty || that instanceof ColGroupConst;
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		return new MatrixBlock(0, 0, false);
	}

	@Override
	public void tsmm(MatrixBlock ret) {
		// do nothing
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock c) {
		// do nothing
	}

	@Override
	public boolean isDense() {
		return false;
	}

	@Override
	public AColGroup copy() {
		return new ColGroupEmpty(_colIndexes, _numRows);
	}

	@Override
	public boolean containsValue(double pattern) {
		return pattern == 0;
	}

	@Override
	public long getNumberNonZeros() {
		return 0;
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		return new ColGroupEmpty(new int[] {0}, _numRows);
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		return new ColGroupEmpty(outputCols, _numRows);
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		return null;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		if(pattern == 0)
			return ColGroupFactory.getColGroupConst(getNumRows(), _colIndexes, replace);
		else
			return new ColGroupEmpty(_colIndexes, getNumRows());
	}
}
