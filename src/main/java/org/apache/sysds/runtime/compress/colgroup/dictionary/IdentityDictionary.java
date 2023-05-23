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
import java.lang.ref.SoftReference;
import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * A specialized dictionary that exploits the fact that the contained dictionary is an Identity Matrix.
 */
public class IdentityDictionary extends ADictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	/** The number of rows or columns, rows can be +1 if withEmpty is set. */
	protected final int nRowCol;
	/** Specify if the Identity matrix should contain an empty row in the end. */
	protected final boolean withEmpty;
	/** A Cache to contain a materialized version of the identity matrix. */
	protected SoftReference<MatrixBlockDictionary> cache = null;

	/**
	 * Create an identity matrix dictionary. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol The number of rows and columns in this identity matrix.
	 */
	public IdentityDictionary(int nRowCol) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = false;
	}

	/**
	 * Create an identity matrix dictionary, It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol   The number of rows and columns in this identity matrix.
	 * @param withEmpty If the matrix should contain an empty row in the end.
	 */
	public IdentityDictionary(int nRowCol, boolean withEmpty) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = withEmpty;
	}

	@Override
	public double[] getValues() {
		throw new DMLCompressionException("Invalid to materialize identity Matrix Please Implement alternative");
		// LOG.warn("Should not call getValues on Identity Dictionary");
		// double[] ret = new double[nRowCol * nRowCol];
		// for(int i = 0; i < nRowCol; i++) {
		// ret[(i * nRowCol) + i] = 1;
		// }
		// return ret;
	}

	@Override
	public double getValue(int i) {
		final int nCol = nRowCol;
		final int row = i / nCol;
		if(row > nRowCol)
			return 0;
		final int col = i % nCol;
		return row == col ? 1 : 0;
	}

	@Override
	public double getValue(int r, int c, int nCol) {
		return r == c ? 1 : 0;
	}

	@Override
	public long getInMemorySize() {
		return 4 + 4 + 8; // int + padding + softReference
	}

	public static long getInMemorySize(int numberColumns) {
		return 4 + 4 + 8;
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		if(fn.getBuiltinCode() == BuiltinCode.MAX)
			return fn.execute(init, 1);
		else if(fn.getBuiltinCode() == BuiltinCode.MIN)
			return fn.execute(init, 0);
		else
			throw new NotImplementedException();
	}

	@Override
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		return getMBDict().aggregateWithReference(init, fn, reference, def);
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, fn.execute(1, 0));
		return ret;
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
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		for(int i = 0; i < nRowCol; i++) {
			final int idx = colIndexes.get(i);
			c[idx] = fn.execute(c[idx], 0);
			c[idx] = fn.execute(c[idx], 1);
		}
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, IColIndex colIndexes, double[] reference,
		boolean def) {
		getMBDict().aggregateColsWithReference(c, fn, colIndexes, reference, def);
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op) {
		return getMBDict().applyScalarOp(op);
	}

	@Override
	public ADictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {

		return getMBDict().applyScalarOpAndAppend(op, v0, nCol);
	}

	@Override
	public ADictionary applyUnaryOp(UnaryOperator op) {
		return getMBDict().applyUnaryOp(op);
	}

	@Override
	public ADictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		return getMBDict().applyUnaryOpAndAppend(op, v0, nCol);
	}

	@Override
	public ADictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
		return getMBDict().applyScalarOpWithReference(op, reference, newReference);
	}

	@Override
	public ADictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
		return getMBDict().applyUnaryOpWithReference(op, reference, newReference);
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpLeft(op, v, colIndexes);
	}

	@Override
	public ADictionary binOpLeftAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpLeftAndAppend(op, v, colIndexes);
	}

	@Override
	public ADictionary binOpLeftWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		return getMBDict().binOpLeftWithReference(op, v, colIndexes, reference, newReference);

	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpRight(op, v, colIndexes);
	}

	@Override
	public ADictionary binOpRightAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpRightAndAppend(op, v, colIndexes);
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v) {
		return getMBDict().binOpRight(op, v);
	}

	@Override
	public ADictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		return getMBDict().binOpRightWithReference(op, v, colIndexes, reference, newReference);
	}

	@Override
	public ADictionary clone() {
		return new IdentityDictionary(nRowCol);
	}

	@Override
	public DictType getDictType() {
		return DictType.Identity;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return nRowCol + (withEmpty ? 1 : 0);
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, 1);
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, 1);
		for(int i = 0; i < defaultTuple.length; i++)
			ret[i] += defaultTuple[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, 1);
		for(int i = 0; i < reference.length; i++)
			ret[i] += reference[i] * nRowCol;
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, 1);
		return ret;
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
	public double[] productAllRowsToDouble(int nCol) {
		return new double[nRowCol];
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		return new double[nRowCol];
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		return getMBDict().productAllRowsToDoubleWithReference(reference);
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		for(int i = 0; i < colIndexes.size(); i++) {
			// very nice...
			final int idx = colIndexes.get(i);
			c[idx] = counts[i];
		}
	}

	@Override
	public void colSumSq(double[] c, int[] counts, IColIndex colIndexes) {
		colSum(c, counts, colIndexes);
	}

	@Override
	public void colProduct(double[] res, int[] counts, IColIndex colIndexes) {
		for(int i = 0; i < colIndexes.size(); i++) {
			res[colIndexes.get(i)] = 0;
		}
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
	public double sum(int[] counts, int ncol) {
		// number of rows, change this.
		double s = 0.0;
		for(int v : counts)
			s += v;
		return s;
	}

	@Override
	public double sumSq(int[] counts, int ncol) {
		return sum(counts, ncol);
	}

	@Override
	public double sumSqWithReference(int[] counts, double[] reference) {
		return getMBDict().sumSqWithReference(counts, reference);
	}

	@Override
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		if(idxStart == 0 && idxEnd == nRowCol)
			return new IdentityDictionary(nRowCol);
		else
			return new IdentityDictionarySlice(nRowCol, idxStart, idxEnd);
	}

	@Override
	public boolean containsValue(double pattern) {
		return pattern == 0.0 || pattern == 1.0;
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		return getMBDict().containsValueWithReference(pattern, reference);
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		return (long) sum(counts, nCol);
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		return getMBDict().getNumberNonZerosWithReference(counts, reference, nRows);
	}

	@Override
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol) {
		getMBDict().addToEntry(v, fr, to, nCol);
	}

	@Override
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol, int rep) {
		getMBDict().addToEntry(v, fr, to, nCol, rep);
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		getMBDict().addToEntryVectorized(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		return getMBDict().subtractTuple(tuple);
	}

	public MatrixBlockDictionary getMBDict() {
		return getMBDict(nRowCol);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		if(cache != null) {
			MatrixBlockDictionary r = cache.get();
			if(r != null)
				return r;
		}
		MatrixBlockDictionary ret = createMBDict();
		cache = new SoftReference<>(ret);
		return ret;
	}

	private MatrixBlockDictionary createMBDict() {
		if(withEmpty) {
			final SparseBlock sb = SparseBlockFactory.createIdentityMatrixWithEmptyRow(nRowCol);
			final MatrixBlock identity = new MatrixBlock(nRowCol + 1, nRowCol, nRowCol, sb);
			return new MatrixBlockDictionary(identity);
		}
		else {

			final SparseBlock sb = SparseBlockFactory.createIdentityMatrix(nRowCol);
			final MatrixBlock identity = new MatrixBlock(nRowCol, nRowCol, nRowCol, sb);
			return new MatrixBlockDictionary(identity);
		}
	}

	@Override
	public String getString(int colIndexes) {
		return "IdentityMatrix of size: " + nRowCol;
	}

	@Override
	public String toString() {
		return "IdentityMatrix of size: " + nRowCol;
	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
		return getMBDict().scaleTuples(scaling, nCol);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.IDENTITY.ordinal());
		out.writeInt(nRowCol);
	}

	public static IdentityDictionary read(DataInput in) throws IOException {
		return new IdentityDictionary(in.readInt());
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4;
	}

	@Override
	public ADictionary preaggValuesFromDense(final int numVals, final IColIndex colIndexes,
		final IColIndex aggregateColumns, final double[] b, final int cut) {
		return getMBDict().preaggValuesFromDense(numVals, colIndexes, aggregateColumns, b, cut);
	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		if(containsValue(pattern))
			return getMBDict().replace(pattern, replace, nCol);
		else
			return this;
	}

	@Override
	public ADictionary replaceWithReference(double pattern, double replace, double[] reference) {
		if(containsValueWithReference(pattern, reference))
			return getMBDict().replaceWithReference(pattern, replace, reference);
		else
			return this;
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
	public CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows) {
		return getMBDict().centralMoment(ret, fn, counts, nRows);
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		return getMBDict().centralMomentWithDefault(ret, fn, counts, def, nRows);
	}

	@Override
	public CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		return getMBDict().centralMomentWithReference(ret, fn, counts, reference, nRows);
	}

	@Override
	public ADictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		return getMBDict().rexpandCols(max, ignore, cast, nCol);
	}

	@Override
	public ADictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		return getMBDict().rexpandColsWithReference(max, ignore, cast, reference);
	}

	@Override
	public double getSparsity() {
		return 1.0d / (double) nRowCol;
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		getMBDict().multiplyScalar(v, ret, off, dictIdx, cols);
	}

	@Override
	protected void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		getMBDict().TSMMWithScaling(counts, rows, cols, ret);
	}

	@Override
	protected void MMDict(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDict(right, rowsLeft, colsRight, result);
		// should replace with add to right to output cells.
	}

	@Override
	protected void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDictDense(left, rowsLeft, colsRight, result);
		// should replace with add to right to output cells.
	}

	@Override
	protected void MMDictSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDictSparse(left, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangle(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().TSMMToUpperTriangle(right, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleDense(left, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleSparse(left, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleScaling(ADictionary right, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleScaling(right, rowsLeft, colsRight, scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleDenseScaling(double[] left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		getMBDict().TSMMToUpperTriangleDenseScaling(left, rowsLeft, colsRight, scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparseScaling(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight,
		int[] scale, MatrixBlock result) {

		getMBDict().TSMMToUpperTriangleSparseScaling(left, rowsLeft, colsRight, scale, result);
	}

	@Override
	public boolean equals(ADictionary o) {
		if(o instanceof IdentityDictionary)
			return ((IdentityDictionary) o).nRowCol == nRowCol;

		MatrixBlock mb = getMBDict().getMatrixBlock();
		if(o instanceof MatrixBlockDictionary)
			return mb.equals(((MatrixBlockDictionary) o).getMatrixBlock());
		else if(o instanceof Dictionary) {
			if(mb.isInSparseFormat())
				return mb.getSparseBlock().equals(((Dictionary) o)._values, nRowCol);
			final double[] dv = mb.getDenseBlockValues();
			return Arrays.equals(dv, ((Dictionary) o)._values);
		}

		return false;
	}

	@Override
	public ADictionary cbind(ADictionary that, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary reorder(int[] reorder) {
		return getMBDict().reorder(reorder);
	}

}
