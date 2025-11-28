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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * Base class for column groups encoded Encoded in a compressed manner.
 */
public abstract class AColGroupCompressed extends AColGroup {

	private static final long serialVersionUID = 6219835795420081223L;

	protected AColGroupCompressed() {
		super();
	}

	protected AColGroupCompressed(IColIndex colIndices) {
		super(colIndices);
	}

	protected abstract double computeMxx(double c, Builtin builtin);

	protected abstract void computeColMxx(double[] c, Builtin builtin);

	protected abstract void computeSum(double[] c, int nRows);

	protected abstract void computeSumSq(double[] c, int nRows);

	protected abstract void computeColSumsSq(double[] c, int nRows);

	/**
	 * Compute row sums, note that this function works even for row SQ, since the preaggregate is correct.
	 * 
	 * @param c      target to aggregate into
	 * @param rl     row to start from
	 * @param ru     row to end at (not inclusive)
	 * @param preAgg the pre-aggregated rows from this column group.
	 */
	protected abstract void computeRowSums(double[] c, int rl, int ru, double[] preAgg);

	protected abstract void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg);

	protected abstract void computeProduct(double[] c, int nRows);

	protected abstract void computeRowProduct(double[] c, int rl, int ru, double[] preAgg);

	protected abstract void computeColProduct(double[] c, int nRows);

	protected abstract double[] preAggSumRows();

	protected abstract double[] preAggSumSqRows();

	protected abstract double[] preAggProductRows();

	protected abstract double[] preAggBuiltinRows(Builtin builtin);

	@Override
	public boolean sameIndexStructure(AColGroup that) {
		if(that instanceof AColGroupCompressed)
			return sameIndexStructure((AColGroupCompressed) that);
		else
			return false;
	}

	public abstract boolean sameIndexStructure(AColGroupCompressed that);

	public double[] preAggRows(ValueFunction fn) {
		// final ValueFunction fn = op.aggOp.increOp.fn;
		if(fn instanceof KahanPlusSq)
			return preAggSumSqRows();
		else if(fn instanceof Plus || fn instanceof KahanPlus)
			return preAggSumRows();
		else if(fn instanceof Multiply)
			return preAggProductRows();
		else if(fn instanceof Builtin) {
			Builtin bop = (Builtin) fn;
			BuiltinCode bopC = bop.getBuiltinCode();
			if(bopC == BuiltinCode.MAX || bopC == BuiltinCode.MIN)
				return preAggBuiltinRows(bop);
			else
				throw new DMLScriptException("unsupported builtin type: " + bop);
		}
		else
			throw new DMLScriptException("Row Aggregate ValueFunction operator on CompressedMatrixBlock " + fn);
	}

	@Override
	public double getMin() {
		return computeMxx(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MIN));
	}

	@Override
	public double getMax() {
		return computeMxx(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
	}

	@Override
	public double getSum(int nRows) {
		double[] ret = new double[1];
		computeSum(ret, nRows);
		return ret[0];
	}

	@Override
	public final void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int nRows, int rl, int ru) {
		unaryAggregateOperations(op, c, nRows, rl, ru,
			(op.indexFn instanceof ReduceCol) ? preAggRows(op.aggOp.increOp.fn) : null);
	}

	public final void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int nRows, int rl, int ru,
		double[] preAgg) {
		final ValueFunction fn = op.aggOp.increOp.fn;
		if(fn instanceof KahanPlusSq)
			sumSq(op.indexFn, c, nRows, rl, ru, preAgg);
		else if(fn instanceof Plus || fn instanceof KahanPlus)
			sum(op.indexFn, c, nRows, rl, ru, preAgg);
		else if(fn instanceof Multiply)
			prod(op.indexFn, c, nRows, rl, ru, preAgg);
		else if(fn instanceof Builtin)
			builtin(op, c, nRows, rl, ru, preAgg);
		else
			throw new DMLRuntimeException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
	}

	private final void sumSq(IndexFunction idx, double[] c, int nRows, int rl, int ru, double[] preAgg) {
		if(idx instanceof ReduceAll)
			computeSumSq(c, nRows);
		else if(idx instanceof ReduceCol) // This call works becasuse the preAgg is correctly the sumsq.
			computeRowSums(c, rl, ru, preAgg);
		else if(idx instanceof ReduceRow)
			computeColSumsSq(c, nRows);
		else
			throw new DMLRuntimeException("unsupported index type in colgroup: " + idx);
	}

	private final void sum(IndexFunction idx, double[] c, int nRows, int rl, int ru, double[] preAgg) {
		if(idx instanceof ReduceAll)
			computeSum(c, nRows);
		else if(idx instanceof ReduceCol)
			computeRowSums(c, rl, ru, preAgg);
		else if(idx instanceof ReduceRow)
			computeColSums(c, nRows);
		else
			throw new DMLRuntimeException("unsupported index type in colgroup: " + idx);
	}

	private final void prod(IndexFunction idx, double[] c, int nRows, int rl, int ru, double[] preAgg) {
		if(idx instanceof ReduceAll)
			computeProduct(c, nRows);
		else if(idx instanceof ReduceCol)
			computeRowProduct(c, rl, ru, preAgg);
		else if(idx instanceof ReduceRow)
			computeColProduct(c, nRows);
		else
			throw new DMLRuntimeException("unsupported index type in colgroup: " + idx);
	}

	private final void builtin(AggregateUnaryOperator op, double[] c, int nRows, int rl, int ru, double[] preAgg) {
		Builtin bop = (Builtin) op.aggOp.increOp.fn;
		BuiltinCode bopC = bop.getBuiltinCode();
		if(bopC == BuiltinCode.MAX || bopC == BuiltinCode.MIN) {
			if(op.indexFn instanceof ReduceAll)
				c[0] = computeMxx(c[0], bop);
			else if(op.indexFn instanceof ReduceCol)
				computeRowMxx(c, bop, rl, ru, preAgg);
			else if(op.indexFn instanceof ReduceRow)
				computeColMxx(c, bop);
			else
				throw new DMLRuntimeException("unsupported index type in colgroup: " + op.indexFn);
		}
		else
			throw new DMLRuntimeException("unsupported builtin type: " + bop);
	}

	@Override
	public final void tsmm(MatrixBlock ret, int nRows) {
		double[] result = ret.getDenseBlockValues();
		int numColumns = ret.getNumColumns();
		tsmm(result, numColumns, nRows);
	}

	protected abstract void tsmm(double[] result, int numColumns, int nRows);

	protected static void tsmm(double[] result, int numColumns, int[] counts, IDictionary dict, IColIndex colIndexes) {
		dict = dict.getMBDict(colIndexes.size());
		final MatrixBlock mb = ((MatrixBlockDictionary) dict).getMatrixBlock();
		if(mb.isInSparseFormat())
			tsmmSparse(result, numColumns, mb.getSparseBlock(), counts, colIndexes);
		else
			tsmmDense(result, numColumns, mb.getDenseBlockValues(), counts, colIndexes);

	}

	protected static void tsmmDense(double[] result, int numColumns, double[] values, int[] counts,
		IColIndex colIndexes) {
		final int nCol = colIndexes.size();
		final int nRow = counts.length;
		for(int k = 0; k < nRow; k++) {
			final int offTmp = nCol * k;
			final int scale = counts[k];
			for(int i = 0; i < nCol; i++) {
				final int offRet = numColumns * colIndexes.get(i);
				final double v = values[offTmp + i] * scale;
				if(v != 0)
					for(int j = i; j < nCol; j++)
						result[offRet + colIndexes.get(j)] += v * values[offTmp + j];
			}
		}
	}

	protected static void tsmmSparse(double[] result, int numColumns, SparseBlock sb, int[] counts,
		IColIndex colIndexes) {
		for(int row = 0; row < counts.length; row++) {
			if(sb.isEmpty(row))
				continue;
			final int apos = sb.pos(row);
			final int alen = sb.size(row);
			final int[] aix = sb.indexes(row);
			final double[] avals = sb.values(row);
			for(int i = apos; i < apos + alen; i++) {
				final int offRet = colIndexes.get(aix[i]) * numColumns;
				final double val = avals[i] * counts[row];
				for(int j = i; j < apos + alen; j++) {
					result[offRet + colIndexes.get(aix[j])] += val * avals[j];
				}
			}
		}
	}

	@Override
	public boolean isEmpty() {
		return false;
	}
}
