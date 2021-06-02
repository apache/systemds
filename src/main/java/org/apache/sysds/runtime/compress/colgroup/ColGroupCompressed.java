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
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Plus;
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

	public abstract double[] getValues();

	public abstract void addMinMax(double[] ret);

	public abstract boolean isLossy();

	protected abstract double computeMxx(double c, Builtin builtin);

	protected abstract void computeColMxx(double[] c, Builtin builtin);

	protected abstract void computeSum(double[] c, boolean square);

	protected abstract void computeRowSums(double[] c, boolean square, int rl, int ru);

	protected abstract void computeColSums(double[] c, boolean square);

	protected abstract void computeRowMxx(double[] c, Builtin builtin, int rl, int ru);

	protected abstract boolean sameIndexStructure(ColGroupCompressed that);

	@Override
	public final double getMin() {
		return computeMxx(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MIN));
	}

	@Override
	public final double getMax() {
		return computeMxx(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
	}

	@Override
	public final void unaryAggregateOperations(AggregateUnaryOperator op, double[] c) {
		unaryAggregateOperations(op, c, 0, _numRows);
	}

	@Override
	public final void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int rl, int ru) {
		if(op.aggOp.increOp.fn instanceof Plus || op.aggOp.increOp.fn instanceof KahanPlus ||
			op.aggOp.increOp.fn instanceof KahanPlusSq) {
			boolean square = op.aggOp.increOp.fn instanceof KahanPlusSq;
			if(op.indexFn instanceof ReduceAll)
				computeSum(c, square);
			else if(op.indexFn instanceof ReduceCol)
				computeRowSums(c, square, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColSums(c, square);
		}
		else if(op.aggOp.increOp.fn instanceof Builtin) {
			Builtin bop = (Builtin) op.aggOp.increOp.fn;
			BuiltinCode bopC = bop.getBuiltinCode();
			if(bopC == BuiltinCode.MAX || bopC == BuiltinCode.MIN) {
				if(op.indexFn instanceof ReduceAll)
					c[0] = computeMxx(c[0], bop);
				else if(op.indexFn instanceof ReduceCol)
					computeRowMxx(c, bop, rl, ru);
				else if(op.indexFn instanceof ReduceRow)
					computeColMxx(c, bop);
			}
			else {
				throw new DMLScriptException("unsupported builtin type: " + bop);
			}

		}
		else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" num Rows: " + getNumRows());
		sb.append(super.toString());
		return sb.toString();
	}

	@Override
	public final int getNumRows() {
		return _numRows;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += 4;
		return size;
	}
}
