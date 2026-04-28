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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.io.Serializable;

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique tuple values of a column group.
 */
public abstract class ADictionary implements IDictionary, Serializable {
	private static final long serialVersionUID = 9118692576356558592L;

	public abstract IDictionary clone();

	public final CmCovObject centralMoment(ValueFunction fn, int[] counts, int nRows) {
		return centralMoment(new CmCovObject(), fn, counts, nRows);
	}

	public final CmCovObject centralMomentWithDefault(ValueFunction fn, int[] counts, double def, int nRows) {
		return centralMomentWithDefault(new CmCovObject(), fn, counts, def, nRows);
	}

	public final CmCovObject centralMomentWithReference(ValueFunction fn, int[] counts, double reference, int nRows) {
		return centralMomentWithReference(new CmCovObject(), fn, counts, reference, nRows);
	}

	@Override
	public final boolean equals(Object o) {
		if(o != null && o instanceof IDictionary)
			return equals((IDictionary) o);
		return false;
	}

	@Override
	public final boolean equals(double[] v) {
		return equals(new Dictionary(v));
	}

	/**
	 * Make a double into a string, if the double is a whole number then return it without decimal points
	 * 
	 * @param v The value
	 * @return The string
	 */
	protected static String doubleToString(double v) {
		if(v == (long) v)
			return Long.toString(((long) v));
		else
			return Double.toString(v);
	}

	/**
	 * Correct Nan Values in an result. If there are any NaN values in the given Res then they are replaced with 0.
	 * 
	 * @param res        The array to correct
	 * @param colIndexes The column indexes.
	 */
	public static void correctNan(double[] res, IColIndex colIndexes) {
		// since there is no nan values in most dictionaries, we exploit that
		// nan only occur if we multiplied infinity with 0.
		for(int j = 0; j < colIndexes.size(); j++) {
			final int cix = colIndexes.get(j);
			res[cix] = Double.isNaN(res[cix]) ? 0 : res[cix];
		}
	}

	@Override
	public IDictionary rightMMPreAggSparse(int numVals, SparseBlock b, IColIndex thisCols, IColIndex aggregateColumns,
		int nColRight) {
		if(aggregateColumns.size() < nColRight)
			return rightMMPreAggSparseSelectedCols(numVals, b, thisCols, aggregateColumns);
		else
			return rightMMPreAggSparseAllColsRight(numVals, b, thisCols, nColRight);
	}

	@Override
	public void putSparse(SparseBlock sb, int idx, int rowOut, int nCol, IColIndex columns) {
		for(int i = 0; i < nCol; i++)
			sb.append(rowOut, columns.get(i), getValue(idx, i, nCol));
	}

	@Override
	public void putDense(DenseBlock dr, int idx, int rowOut, int nCol, IColIndex columns) {
		double[] dv = dr.values(rowOut);
		int off = dr.pos(rowOut);
		for(int i = 0; i < nCol; i++)
			dv[off + columns.get(i)] += getValue(idx, i, nCol);
	}

	@Override
	public double[] getRow(int i, int nCol) {
		double[] ret = new double[nCol];
		for(int c = 0; c < nCol; c++) {
			ret[c] = getValue(i, c, nCol);
		}
		return ret;
	}

	public MatrixBlockDictionary getMBDict() {
		throw new RuntimeException("Invalid call to getMBDict for " + getClass().getSimpleName());
	}

	@Override
	public void product(double[] ret, int[] counts, int nCol) {
		getMBDict().product(ret, counts, nCol);
	}

	@Override
	public void productWithDefault(double[] ret, int[] counts, double[] def, int defCount) {
		getMBDict().productWithDefault(ret, counts, def, defCount);
	}

	@Override
	public void productWithReference(double[] ret, int[] counts, double[] reference, int refCount) {
		getMBDict().productWithReference(ret, counts, reference, refCount);
	}

	@Override
	public CmCovObject centralMoment(CmCovObject ret, ValueFunction fn, int[] counts, int nRows) {
		return getMBDict().centralMoment(ret, fn, counts, nRows);
	}

	@Override
	public double getSparsity() {
		return getMBDict().getSparsity();
	}

	@Override
	public CmCovObject centralMomentWithDefault(CmCovObject ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		return getMBDict().centralMomentWithDefault(ret, fn, counts, def, nRows);
	}

	@Override
	public CmCovObject centralMomentWithReference(CmCovObject ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		return getMBDict().centralMomentWithReference(ret, fn, counts, reference, nRows);
	}

	@Override
	public IDictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		return getMBDict().rexpandCols(max, ignore, cast, nCol);
	}

	@Override
	public IDictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		return getMBDict().rexpandColsWithReference(max, ignore, cast, reference);
	}

	@Override
	public void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		getMBDict().TSMMWithScaling(counts, rows, cols, ret);
	}

	@Override
	public void MMDict(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDict(right, rowsLeft, colsRight, result);
	}

	@Override
	public void MMDictScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		getMBDict().MMDictScaling(right, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void MMDictSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDictSparse(left, rowsLeft, colsRight, result);
	}

	@Override
	public void MMDictScalingSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		getMBDict().MMDictScalingSparse(left, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void TSMMToUpperTriangle(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().TSMMToUpperTriangle(right, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleDense(left, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleSparse(left, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleScaling(right, rowsLeft, colsRight, scale, result);
	}

	@Override
	public void TSMMToUpperTriangleDenseScaling(double[] left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleDenseScaling(left, rowsLeft, colsRight, scale, result);
	}

	@Override
	public void TSMMToUpperTriangleSparseScaling(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleSparseScaling(left, rowsLeft, colsRight, scale, result);
	}

	@Override
	public IDictionary reorder(int[] reorder) {
		return getMBDict().reorder(reorder);
	}

	@Override
	public IDictionary cbind(IDictionary that, int nCol) {
		return getMBDict().cbind(that, nCol);
	}

	@Override
	public IDictionary append(double[] row) {
		return getMBDict().append(row);
	}

	@Override
	public IDictionary replace(double pattern, double replace, int nCol) {
		if(containsValue(pattern))
			return getMBDict().replace(pattern, replace, nCol);
		else
			return this;
	}

	@Override
	public IDictionary replaceWithReference(double pattern, double replace, double[] reference) {
		if(containsValueWithReference(pattern, reference))
			return getMBDict().replaceWithReference(pattern, replace, reference);
		else
			return this;
	}

	@Override
	public IDictionary subtractTuple(double[] tuple) {
		return getMBDict().subtractTuple(tuple);
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		return getMBDict().getNumberNonZerosWithReference(counts, reference, nRows);
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		if(Double.isNaN(pattern)) {
			for(int i = 0; i < reference.length; i++)
				if(Double.isNaN(reference[i]))
					return true;
			return containsValue(pattern);
		}
		return getMBDict().containsValueWithReference(pattern, reference);
	}

	@Override
	public double sumSqWithReference(int[] counts, double[] reference) {
		return getMBDict().sumSqWithReference(counts, reference);
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, IColIndex colIndexes, double[] reference) {
		getMBDict().colProductWithReference(res, counts, colIndexes, reference);

	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, IColIndex colIndexes, double[] reference) {
		getMBDict().colSumSqWithReference(c, counts, colIndexes, reference);
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		getMBDict().multiplyScalar(v, ret, off, dictIdx, cols);
	}

	@Override
	public void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDictDense(left, rowsLeft, colsRight, result);
	}

	protected IDictionary rightMMPreAggSparseAllColsRight(int numVals, SparseBlock b, IColIndex thisCols,
		int nColRight) {
		return getMBDict().rightMMPreAggSparseAllColsRight(numVals, b, thisCols, nColRight);
	}

	protected IDictionary rightMMPreAggSparseSelectedCols(int numVals, SparseBlock b, IColIndex thisCols,
		IColIndex aggregateColumns) {
		return getMBDict().rightMMPreAggSparseSelectedCols(numVals, b, thisCols, aggregateColumns);
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		return getMBDict().productAllRowsToDoubleWithReference(reference);
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple) {
		return getMBDict().sumAllRowsToDoubleSqWithDefault(defaultTuple);
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithReference(double[] reference) {
		return getMBDict().sumAllRowsToDoubleSqWithReference(reference);
	}

	@Override
	public IDictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		return getMBDict().binOpRightWithReference(op, v, colIndexes, reference, newReference);
	}

	@Override
	public IDictionary binOpRightAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpRightAndAppend(op, v, colIndexes);
	}

	@Override
	public IDictionary binOpRight(BinaryOperator op, double[] v) {
		return getMBDict().binOpRight(op, v);
	}

	@Override
	public IDictionary applyScalarOp(ScalarOperator op) {
		return getMBDict().applyScalarOp(op);
	}

	@Override
	public IDictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
		return getMBDict().applyScalarOpAndAppend(op, v0, nCol);
	}

	@Override
	public IDictionary applyUnaryOp(UnaryOperator op) {
		return getMBDict().applyUnaryOp(op);
	}

	@Override
	public IDictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		return getMBDict().applyUnaryOpAndAppend(op, v0, nCol);
	}

	@Override
	public IDictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
		return getMBDict().applyScalarOpWithReference(op, reference, newReference);
	}

	@Override
	public IDictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
		return getMBDict().applyUnaryOpWithReference(op, reference, newReference);
	}

	@Override
	public IDictionary binOpLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpLeft(op, v, colIndexes);
	}

	@Override
	public IDictionary binOpLeftAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpLeftAndAppend(op, v, colIndexes);
	}

	@Override
	public IDictionary binOpLeftWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		return getMBDict().binOpLeftWithReference(op, v, colIndexes, reference, newReference);
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, IColIndex colIndexes, double[] reference,
		boolean def) {
		getMBDict().aggregateColsWithReference(c, fn, colIndexes, reference, def);
	}

	@Override
	public double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple) {
		return getMBDict().aggregateRowsWithDefault(fn, defaultTuple);
	}

	@Override
	public double[] aggregateRowsWithReference(Builtin fn, double[] reference) {
		return getMBDict().aggregateRowsWithReference(fn, reference);
	}

	@Override
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		return getMBDict().aggregateWithReference(init, fn, reference, def);
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		return getMBDict().aggregate(init, fn);
	}

	@Override
	public void colSumSq(double[] c, int[] counts, IColIndex colIndexes) {
		getMBDict().colSumSq(c, counts, colIndexes);
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol) {
		getMBDict().addToEntry(v, fr, to, nCol);
	}

	@Override
	public void colProduct(double[] res, int[] counts, IColIndex colIndexes) {
		getMBDict().colProduct(res, counts, colIndexes);
	}

	@Override
	public void MMDictScalingDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		getMBDict().MMDictScalingDense(left, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public int[] countNNZZeroColumns(int[] counts) {
		return getMBDict().countNNZZeroColumns(counts);
	}

	@Override
	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		return getMBDict().sliceOutColumnRange(idxStart, idxEnd, previousNumberOfColumns);
	}

	@Override
	public IDictionary scaleTuples(int[] scaling, int nCol) {
		return getMBDict().scaleTuples(scaling, nCol);
	}

	@Override
	public IDictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpRight(op, v, colIndexes);
	}

	@Override
	public IDictionary preaggValuesFromDense(final int numVals, final IColIndex colIndexes,
		final IColIndex aggregateColumns, final double[] b, final int cut) {
		return getMBDict().preaggValuesFromDense(numVals, colIndexes, aggregateColumns, b, cut);
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		getMBDict().addToEntryVectorized(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
	}

	@Override
	public double[] getValues() {
		return getMBDict().getValues();
	}

	@Override
	public double getValue(int i) {
		return getMBDict().getValue(i);
	}

	@Override
	public double getValue(int r, int col, int nCol) {
		return getMBDict().getValue(r, col, nCol);
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		return getMBDict().aggregateRows(fn, nCol);
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		getMBDict().aggregateCols(c, fn, colIndexes);
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		return getMBDict().sumAllRowsToDouble(nrColumns);
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		return getMBDict().sumAllRowsToDoubleWithDefault(defaultTuple);
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		return getMBDict().sumAllRowsToDoubleWithReference(reference);
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		return getMBDict().sumAllRowsToDoubleSq(nrColumns);
	}

	@Override
	public double[] productAllRowsToDouble(int nrColumns) {
		return getMBDict().productAllRowsToDouble(nrColumns);
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		return getMBDict().productAllRowsToDoubleWithDefault(defaultTuple);
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		getMBDict().colSum(c, counts, colIndexes);
	}

	@Override
	public double sum(int[] counts, int nCol) {
		return getMBDict().sum(counts, nCol);
	}

	@Override
	public double sumSq(int[] counts, int nCol) {
		return getMBDict().sumSq(counts, nCol);
	}

	@Override
	public boolean containsValue(double pattern) {
		return getMBDict().containsValue(pattern);
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol, int rep) {
		getMBDict().addToEntry(v, fr, to, nCol, rep);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		return getMBDict();
	}

}
