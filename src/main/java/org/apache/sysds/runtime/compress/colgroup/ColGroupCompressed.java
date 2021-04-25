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
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
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

	/**
	 * Slice out the column given, if the column is not contained in the colGroup return null.
	 * 
	 * @param col The column to slice.
	 * @return Either a new colGroup or null.
	 */
	protected abstract AColGroup sliceSingleColumn(int col);

	protected abstract AColGroup sliceMultiColumns(int cl, int cu);

	protected abstract boolean sameIndexStructure(ColGroupCompressed that);

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
		if(op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq ||
			op.aggOp.increOp.fn instanceof Mean) {
			KahanFunction kplus = (op.aggOp.increOp.fn instanceof KahanPlus ||
				op.aggOp.increOp.fn instanceof Mean) ? KahanPlus
					.getKahanPlusFnObject() : KahanPlusSq.getKahanPlusSqFnObject();
			boolean mean = op.aggOp.increOp.fn instanceof Mean;
			if(op.indexFn instanceof ReduceAll)
				computeSum(c, kplus instanceof KahanPlusSq);
			else if(op.indexFn instanceof ReduceCol)
				computeRowSums(c, kplus instanceof KahanPlusSq, rl, ru, mean);
			else if(op.indexFn instanceof ReduceRow)
				computeColSums(c, kplus instanceof KahanPlusSq);
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
	public AColGroup sliceColumns(int cl, int cu) {
		if(cu - cl == 1)
			return sliceSingleColumn(cl);
		else {
			return sliceMultiColumns(cl, cu);
		}
	}

	@Override
	public int getNumRows() {
		return _numRows;
	}

}
