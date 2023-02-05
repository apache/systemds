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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class IdentityDictionarySlice extends IdentityDictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	private final int l;
	private final int u;

	/**
	 * Create a Identity matrix dictionary slice. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol the number of rows and columns in this identity matrix.
	 * @param l       the index lower to start at
	 * @param u       the index upper to end at (not inclusive)
	 */
	public IdentityDictionarySlice(int nRowCol, int l, int u) {
		super(nRowCol);
		if(u > nRowCol || l < 0 || l >= u)
			throw new DMLRuntimeException("Invalid slice Identity: " + nRowCol + " range: " + l + "--" + u);
		this.l = l;
		this.u = u;
	}

	@Override
	public double[] getValues() {
		LOG.warn("Should not call getValues on Identity Dictionary");
		int nCol = u - l;
		double[] ret = new double[nCol * nRowCol];
		for(int i = l; i < u; i++) {
			ret[(i * nCol) + i] = 1;
		}
		return ret;
	}

	@Override
	public double getValue(int i) {
		throw new NotImplementedException();
		// final int nCol = nRowCol;
		// final int row = i / nCol;
		// if(row > nRowCol)
		// return 0;
		// final int col = i % nCol;
		// return row == col ? 1 : 0;
	}

	@Override
	public final double getValue(int r, int c, int nCol) {
		throw new NotImplementedException();
		// return r == c ? 1 : 0;
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(nRowCol);
	}

	public static long getInMemorySize(int numberColumns) {
		// int * 3 + padding + softReference
		return 12 + 4 + 8;
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, l, u, fn.execute(1, 0));
		return ret;
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		for(int i = 0; i < u - l; i++) {
			final int idx = colIndexes.get(i);
			c[idx] = fn.execute(c[idx], 0);
			c[idx] = fn.execute(c[idx], 1);
		}
	}

	@Override
	public ADictionary clone() {
		return new IdentityDictionarySlice(nRowCol, l, u);
	}

	@Override
	public DictType getDictType() {
		return DictType.IdentitySlice;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return ncol;
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, l, u, 1.0);
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, l, u, 1.0);
		for(int i = 0; i < defaultTuple.length; i++)
			ret[i] += defaultTuple[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, l, u, 1.0);
		for(int i = 0; i < reference.length; i++)
			ret[i] += reference[i] * nRowCol;
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, l, u, 1);
		return ret;
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
	public double sum(int[] counts, int ncol) {
		// number of rows, change this.
		double s = 0.0;
		for(int i = l; i < u; i++)
			s += counts[i];
		return s;
	}

	@Override
	public double sumSq(int[] counts, int ncol) {
		return sum(counts, ncol);
	}


	@Override
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		throw new NotImplementedException("Slice of identity slice ??? this is getting a bit ridiculous");
	}

	@Override
	public boolean containsValue(double pattern) {
		return pattern == 0.0 || pattern == 1.0;
	}


	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		return (long) sum(counts, nCol);
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
		MatrixBlock identity = new MatrixBlock(nRowCol, u - l, true);
		for(int i = l; i < u; i++)
			identity.quickSetValue(i, i, 1.0);

		return new MatrixBlockDictionary(identity);
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

}
