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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

import java.io.DataOutput;
import java.io.IOException;

/**
 * This dictionary class is a specialization for the DeltaDDCColgroup. Here the adjustments for operations for the delta
 * encoded values are implemented.
 */
public class DeltaDictionary extends ADictionary {

	private static final long serialVersionUID = -5700139221491143705L;
	
	private final int _numCols;

	protected final double[] _values;

	public DeltaDictionary(double[] values, int numCols) {
		_values = values;
		_numCols = numCols;
	}

	@Override
	public double[] getValues() {
		throw new NotImplementedException();
	}

	@Override
	public double getValue(int i) {
		throw new NotImplementedException();
	}

	@Override
	public double getValue(int r, int col, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public long getInMemorySize() {
		throw new NotImplementedException();
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		throw new NotImplementedException();
	}

	@Override
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		throw new NotImplementedException();
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple) {
		throw new NotImplementedException();
	}

	@Override
	public double[] aggregateRowsWithReference(Builtin fn, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, IColIndex colIndexes, double[] reference, boolean def) {
		throw new NotImplementedException();
	}

	@Override
	public DeltaDictionary applyScalarOp(ScalarOperator op) {
		final double[] retV = new double[_values.length];
		if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			for(int i = 0; i < _values.length; i++)
				retV[i] = op.executeScalar(_values[i]);
		}
		else if(op.fn instanceof Plus || op.fn instanceof Minus) {
			// With Plus and Minus only the first row needs to be updated when delta encoded
			for(int i = 0; i < _values.length; i++) {
				if(i < _numCols)
					retV[i] = op.executeScalar(_values[i]);
				else
					retV[i] = _values[i];
			}
		}
		else
			throw new NotImplementedException();

		return new DeltaDictionary(retV, _numCols);
	}

	@Override
	public ADictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyUnaryOp(UnaryOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpLeftAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpLeftWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference, double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRightAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference, double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary clone() {
		throw new NotImplementedException();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public long getExactSizeOnDisk() {
		throw new NotImplementedException();
	}

	@Override
	public DictType getDictType() {
		throw new NotImplementedException();
	}

	@Override
	public int getNumberOfValues(int ncol) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple) {
		throw new NotImplementedException();
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithReference(double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] productAllRowsToDouble(int nrColumns) {
		throw new NotImplementedException();
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		throw new NotImplementedException();
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void colSumSq(double[] c, int[] counts, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, IColIndex colIndexes, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double sum(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public double sumSq(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public double sumSqWithReference(int[] counts, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public String getString(int colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		throw new NotImplementedException();
	}

	@Override
	public boolean containsValue(double pattern) {
		throw new NotImplementedException();
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol, int rep) {
		throw new NotImplementedException();
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		throw new NotImplementedException();
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary preaggValuesFromDense(int numVals, IColIndex colIndexes, IColIndex aggregateColumns, double[] b, int cut) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary replaceWithReference(double pattern, double replace, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public void product(double[] ret, int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public void productWithDefault(double[] ret, int[] counts, double[] def, int defCount) {
		throw new NotImplementedException();
	}

	@Override
	public void productWithReference(double[] ret, int[] counts, double[] reference, int refCount) {
		throw new NotImplementedException();
	}

	@Override
	public void colProduct(double[] res, int[] counts, IColIndex colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, IColIndex colIndexes, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts, double reference, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		throw new NotImplementedException();
	}

	@Override
	public double getSparsity() {
		throw new NotImplementedException();
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		throw new NotImplementedException();
	}

	@Override
	protected void MMDict(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void MMDictSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangle(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangleDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangleSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangleScaling(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, int[] scale, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangleDenseScaling(double[] left, IColIndex rowsLeft, IColIndex colsRight, int[] scale, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	protected void TSMMToUpperTriangleSparseScaling(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, int[] scale, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	public boolean equals(ADictionary o) {
		throw new NotImplementedException();
	}
}
