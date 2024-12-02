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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class PlaceHolderDict implements IDictionary, Serializable {

	private static final long serialVersionUID = 9176356558592L;

	private static final String errMessage = "PlaceHolderDict does not support Operations, and is purely intended for serialization";

	/** The number of values supposed to be contained in this dictionary */
	private final int nVal;

	public PlaceHolderDict(int nVal) {
		this.nVal = nVal;
	}

	@Override
	public double[] getValues() {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double getValue(int i) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double getValue(int r, int col, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public long getInMemorySize() {
		return 16 + 4;
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] aggregateRowsWithReference(Builtin fn, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, IColIndex colIndexes, double[] reference,
		boolean def) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyScalarOp(ScalarOperator op) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyUnaryOp(UnaryOperator op) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpLeftAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpLeftWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpRightAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpRight(BinaryOperator op, double[] v) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		byte[] o = new byte[5];
		o[0] = (byte) DictionaryFactory.Type.PLACE_HOLDER.ordinal();
		IOUtilFunctions.intToBa(nVal, o, 1);
		out.write(o);
	}

	public static PlaceHolderDict read(DataInput in) throws IOException {
		int nVals = in.readInt();
		return new PlaceHolderDict(nVals);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4;
	}

	@Override
	public DictType getDictType() {
		throw new RuntimeException(errMessage);
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return nVal;
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithReference(double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] productAllRowsToDouble(int nrColumns) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void colSumSq(double[] c, int[] counts, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, IColIndex colIndexes, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double sum(int[] counts, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double sumSq(int[] counts, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double sumSqWithReference(int[] counts, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public String getString(int colIndexes) {
		return ""; // get string empty
	}

	@Override
	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public boolean containsValue(double pattern) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		return -1;
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void addToEntry(double[] v, int fr, int to, int nCol, int rep) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary subtractTuple(double[] tuple) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary scaleTuples(int[] scaling, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary preaggValuesFromDense(int numVals, IColIndex colIndexes, IColIndex aggregateColumns, double[] b,
		int cut) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary replace(double pattern, double replace, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary replaceWithReference(double pattern, double replace, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void product(double[] ret, int[] counts, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void productWithDefault(double[] ret, int[] counts, double[] def, int defCount) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void productWithReference(double[] ret, int[] counts, double[] reference, int refCount) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void colProduct(double[] res, int[] counts, IColIndex colIndexes) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, IColIndex colIndexes, double[] reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMoment(ValueFunction fn, int[] counts, int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(ValueFunction fn, int[] counts, double def, int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMomentWithReference(ValueFunction fn, int[] counts, double reference, int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public double getSparsity() {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void MMDict(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void MMDictSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangle(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangleDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangleSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangleScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangleDenseScaling(double[] left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void TSMMToUpperTriangleSparseScaling(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary cbind(IDictionary that, int nCol) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public boolean equals(IDictionary o) {
		return o instanceof PlaceHolderDict;
	}

	@Override
	public final boolean equals(double[] v) {
		return false;
	}

	@Override
	public IDictionary reorder(int[] reorder) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public IDictionary clone() {
		return new PlaceHolderDict(nVal);
	}

	@Override
	public void MMDictScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void MMDictScalingDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void MMDictScalingSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		throw new RuntimeException(errMessage);
	}
	
	@Override
	public void putSparse(SparseBlock sb, int idx, int rowOut, int nCol, IColIndex columns) {
		throw new RuntimeException(errMessage);
	}

	@Override
	public void putDense(DenseBlock sb, int idx, int rowOut, int nCol, IColIndex columns) {
		throw new RuntimeException(errMessage);
	}
}
