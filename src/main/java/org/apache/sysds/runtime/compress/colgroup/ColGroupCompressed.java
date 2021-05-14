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

import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * Base class for column groups encoded Encoded in a compressed manner.
 */
public abstract class ColGroupCompressed extends AColGroup {
	private static final long serialVersionUID = 3786247536054353658L;

	final protected int _numRows;

	protected ColGroupCompressed(int numRows) {
		super();
		_numRows = numRows;
	}

	/**
	 * Main constructor for the ColGroupCompresseds. Used to contain the dictionaries used for the different types of
	 * ColGroup.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 * @param cs         The Compression settings used for compression
	 */
	protected ColGroupCompressed(int[] colIndices, int numRows) {
		super(colIndices);
		_numRows = numRows;
	}

	public abstract int getNumValues();

	public abstract double[] getValues();

	public abstract void addMinMax(double[] ret);

	public abstract boolean isLossy();

	/**
	 * if -1 is returned it means false, otherwise it returns an index where the zero tuple can be found.
	 * 
	 * @return A Index where the zero tuple can be found.
	 */
	protected abstract int containsAllZeroTuple();

	protected abstract double computeMxx(double c, Builtin builtin);

	protected abstract void computeColMxx(double[] c, Builtin builtin);

	protected abstract void computeSum(double[] c, boolean square);

	protected abstract void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean);

	protected abstract void computeColSums(double[] c, boolean square);

	protected abstract void computeRowMxx(double[] c, Builtin builtin, int rl, int ru);

	protected abstract boolean sameIndexStructure(ColGroupCompressed that);

	public void leftMultByMatrix(MatrixBlock matrix, double[] result, int numCols, int rl, int ru) {
		if(matrix.isEmpty())
			return;
		else if(matrix.isInSparseFormat())
			leftMultBySparseMatrix(matrix.getSparseBlock(), result, matrix.getNumRows(), numCols, rl, ru);
		else {
			leftMultByMatrix(matrix.getDenseBlockValues(), result, matrix.getNumRows(), numCols, rl, ru);
		}
	}

	/**
	 * Multiply with a matrix on the left.
	 * 
	 * @param matrix  matrix to left multiply
	 * @param result  matrix block result
	 * @param numRows The number of rows in the matrix input
	 * @param numCols The number of columns in the colGroups parent matrix.
	 * @param rl      The row to start the matrix multiplication from
	 * @param ru      The row to stop the matrix multiplication at.
	 */
	public abstract void leftMultByMatrix(double[] matrix, double[] result, int numRows, int numCols, int rl, int ru);

	/**
	 * Multiply with a sparse matrix on the left hand side, and add the values to the output result
	 * 
	 * @param sb      The sparse block to multiply with
	 * @param result  The linearized output matrix
	 * @param numRows The number of rows in the left hand side input matrix (the sparse one)
	 * @param numCols The number of columns in the compression.
	 * @param rl      The row to start the matrix multiplication from
	 * @param ru      The row to stop the matrix multiplication at.
	 */
	public abstract void leftMultBySparseMatrix(SparseBlock sb, double[] result, int numRows, int numCols, int rl,
		int ru);

	@Override
	public double getMin() {
		return computeMxx(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MIN));
	}

	@Override
	public double getMax() {
		return computeMxx(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] c) {
		unaryAggregateOperations(op, c, 0, _numRows);
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int rl, int ru) {
		// sum and sumsq (reduceall/reducerow over tuples and counts)
		if(op.aggOp.increOp.fn instanceof Plus || op.aggOp.increOp.fn instanceof KahanPlus ||
			op.aggOp.increOp.fn instanceof KahanPlusSq) {
			boolean square = op.aggOp.increOp.fn instanceof KahanPlusSq;
			boolean mean = op.aggOp.increOp.fn instanceof Mean;
			if(op.indexFn instanceof ReduceAll)
				computeSum(c, square);
			else if(op.indexFn instanceof ReduceCol)
				computeRowSums(c, square, rl, ru, mean);
			else if(op.indexFn instanceof ReduceRow)
				computeColSums(c, square);
		}
		// min and max (reduceall/reducerow over tuples only)
		else if(op.aggOp.increOp.fn instanceof Builtin &&
			(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX ||
				((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN)) {
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;

			if(op.indexFn instanceof ReduceAll)
				c[0] = computeMxx(c[0], builtin);
			else if(op.indexFn instanceof ReduceCol)
				computeRowMxx(c, builtin, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColMxx(c, builtin);
		}
		else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("num Rows: " + getNumRows());
		return sb.toString();
	}

	@Override
	public int getNumRows() {
		return _numRows;
	}

}
