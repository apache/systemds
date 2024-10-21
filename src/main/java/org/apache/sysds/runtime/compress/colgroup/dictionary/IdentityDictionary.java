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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
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
	protected volatile SoftReference<MatrixBlockDictionary> cache = null;

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
		if(nRowCol < 3) {
			// lets live with it if we call it on 3 columns.
			double[] ret = new double[nRowCol * nRowCol + (withEmpty ? nRowCol : 0)];
			for(int i = 0; i < nRowCol; i++) {
				ret[(i * nRowCol) + i] = 1;
			}
			return ret;
		}
		throw new DMLCompressionException("Invalid to materialize identity Matrix Please Implement alternative");
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

	public boolean withEmpty() {
		return withEmpty;
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
	public IDictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		boolean same = false;
		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			same = true;
			for(int i = 0; i < colIndexes.size() && same; i++)
				same = v[colIndexes.get(i)] == 0.0;
		}
		if(op.fn instanceof Divide) {
			same = true;
			for(int i = 0; i < colIndexes.size() && same; i++)
				same = v[colIndexes.get(i)] == 1.0;
		}
		if(same)
			return this;
		MatrixBlockDictionary mb = getMBDict();
		return mb.binOpRight(op, v, colIndexes);
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
	public IDictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		return getMBDict().binOpRightWithReference(op, v, colIndexes, reference, newReference);
	}

	@Override
	public IDictionary clone() {
		return new IdentityDictionary(nRowCol, withEmpty);
	}

	@Override
	public DictType getDictType() {
		return DictType.Identity;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		if(ncol != nRowCol)
			throw new DMLCompressionException("Invalid call to get Number of values assuming wrong number of columns");
		return nRowCol + (withEmpty ? 1 : 0);
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		if(withEmpty) {
			double[] ret = new double[nRowCol + 1];
			Arrays.fill(ret, 1);
			ret[ret.length - 1] = 0;
			return ret;
		}
		else {
			double[] ret = new double[nRowCol];
			Arrays.fill(ret, 1);
			return ret;
		}
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[getNumberOfValues(defaultTuple.length) + 1];
		for(int i = 0; i < nRowCol; i++)
			ret[i] = 1;

		for(int i = 0; i < defaultTuple.length; i++)
			ret[ret.length - 1] += defaultTuple[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		double[] ret = new double[getNumberOfValues(reference.length)];
		double refSum = 0;
		for(int i = 0; i < reference.length; i++)
			refSum += reference[i];
		Arrays.fill(ret, 1);
		for(int i = 0; i < ret.length; i++)
			ret[i] += refSum;

		if(withEmpty)
			ret[ret.length - 1] += -1;

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
		for(int i = 0; i < colIndexes.size(); i++)
			c[colIndexes.get(i)] += counts[i];
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
		if(withEmpty)
			s -= counts[counts.length - 1];
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
	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		if(idxStart == 0 && idxEnd == nRowCol)
			return new IdentityDictionary(nRowCol, withEmpty);
		else
			return new IdentityDictionarySlice(nRowCol, withEmpty, idxStart, idxEnd);
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
	public int[] countNNZZeroColumns(int[] counts) {
		return counts; // interesting ... but true.
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		return getMBDict().getNumberNonZerosWithReference(counts, reference, nRows);
	}

	@Override
	public final void addToEntry(final double[] v, final int fr, final int to, final int nCol) {
		addToEntry(v, fr, to, nCol, 1);
	}

	@Override
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol, int rep) {
		if(!withEmpty)
			v[to * nCol + fr] += rep;
		else if(fr < nRowCol)
			v[to * nCol + fr] += rep;
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		if(withEmpty)
			addToEntryVectorizedWithEmpty(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
		else
			addToEntryVectorizedNorm(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
	}

	private void addToEntryVectorizedWithEmpty(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7,
		int f8, int t1, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		if(f1 < nRowCol)
			v[t1 * nCol + f1] += 1;
		if(f2 < nRowCol)
			v[t2 * nCol + f2] += 1;
		if(f3 < nRowCol)
			v[t3 * nCol + f3] += 1;
		if(f4 < nRowCol)
			v[t4 * nCol + f4] += 1;
		if(f5 < nRowCol)
			v[t5 * nCol + f5] += 1;
		if(f6 < nRowCol)
			v[t6 * nCol + f6] += 1;
		if(f7 < nRowCol)
			v[t7 * nCol + f7] += 1;
		if(f8 < nRowCol)
			v[t8 * nCol + f8] += 1;
	}

	private void addToEntryVectorizedNorm(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8,
		int t1, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		v[t1 * nCol + f1] += 1;
		v[t2 * nCol + f2] += 1;
		v[t3 * nCol + f3] += 1;
		v[t4 * nCol + f4] += 1;
		v[t5 * nCol + f5] += 1;
		v[t6 * nCol + f6] += 1;
		v[t7 * nCol + f7] += 1;
		v[t8 * nCol + f8] += 1;
	}

	@Override
	public IDictionary subtractTuple(double[] tuple) {
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
	public IDictionary scaleTuples(int[] scaling, int nCol) {
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
	public IDictionary preaggValuesFromDense(final int numVals, final IColIndex colIndexes,
		final IColIndex aggregateColumns, final double[] b, final int cut) {
		/**
		 * This operations is Essentially a Identity matrix multiplication with a right hand side dense matrix, but we
		 * need to slice out the right hand side from the input.
		 *
		 * ColIndexes specify the rows to slice out of the right matrix.
		 *
		 * aggregate columns specify the columns to slice out from the right.
		 */
		final int cs = colIndexes.size();
		final int s = aggregateColumns.size();

		double[] ret = new double[s * numVals];
		int off = 0;
		for(int i = 0; i < cs; i++) {// rows on right
			final int offB = colIndexes.get(i) * cut;
			for(int j = 0; j < s; j++) {
				ret[off++] = b[offB + aggregateColumns.get(j)];
			}
		}

		MatrixBlock db = new MatrixBlock(numVals, s, ret);
		return new MatrixBlockDictionary(db);
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
	public IDictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		return getMBDict().rexpandCols(max, ignore, cast, nCol);
	}

	@Override
	public IDictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		return getMBDict().rexpandColsWithReference(max, ignore, cast, reference);
	}

	@Override
	public double getSparsity() {
		if(withEmpty)
			return 1d / (nRowCol + 1);
		else
			return 1d / nRowCol;
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		if(!withEmpty || dictIdx < nRowCol)
			ret[off + cols.get(dictIdx)] += v;
	}

	@Override
	public void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		getMBDict().TSMMWithScaling(counts, rows, cols, ret);
	}

	@Override
	public void MMDict(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		getMBDict().MMDict(right, rowsLeft, colsRight, result);
	}

	public void MMDictScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		getMBDict().MMDictScaling(right, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		// similar to fused transpose left into right locations.
	
		final int leftSide = rowsLeft.size();
		final int colsOut = result.getNumColumns();
		final int commonDim = Math.min(left.length / leftSide, nRowCol);
		final double[] resV = result.getDenseBlockValues();
		for(int i = 0; i < leftSide; i++) { // rows in left side
			final int offOut = rowsLeft.get(i) * colsOut;
			final int leftOff = i;
			for(int j = 0; j < commonDim; j++) { // cols in left side skipping empty from identity
				resV[offOut + colsRight.get(j)] += left[leftOff + j * leftSide];
			}
		}
	}

	@Override
	public void MMDictScalingDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		// getMBDict().MMDictScalingDense(left, rowsLeft, colsRight, result, scaling);
		final int leftSide = rowsLeft.size();
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int i = 0; i < leftSide; i++) { // rows in left side
			final int offOut = rowsLeft.get(i) * resCols;
			for(int j = 0; j < nRowCol; j++) { // cols in right side
				resV[offOut + colsRight.get(j)] += left[i + j * leftSide] * scaling[j];
			}
		}
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
	public boolean equals(IDictionary o) {
		if(o instanceof IdentityDictionary && //
			((IdentityDictionary) o).nRowCol == nRowCol && //
			((IdentityDictionary) o).withEmpty == withEmpty)
			return true;
		return getMBDict().equals(o);
	}

	@Override
	public IDictionary cbind(IDictionary that, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public IDictionary reorder(int[] reorder) {
		return getMBDict().reorder(reorder);
	}

	@Override
	protected IDictionary rightMMPreAggSparseAllColsRight(int numVals, SparseBlock b, IColIndex thisCols,
		int nColRight) {
		final int thisColsSize = thisCols.size();
		final SparseBlockMCSR ret = new SparseBlockMCSR(numVals);

		for(int h = 0; h < thisColsSize; h++) {
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);

			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;
			for(int i = sPos; i < sEnd; i++) {
				ret.add(h, sIndexes[i], sValues[i]);
			}

		}

		final MatrixBlock retB = new MatrixBlock(numVals, nColRight, -1, ret);
		retB.recomputeNonZeros();
		return MatrixBlockDictionary.create(retB, false);
	}

	@Override
	protected IDictionary rightMMPreAggSparseSelectedCols(int numVals, SparseBlock b, IColIndex thisCols,
		IColIndex aggregateColumns) {

		final int thisColsSize = thisCols.size();
		final int aggColSize = aggregateColumns.size();
		final SparseBlockMCSR ret = new SparseBlockMCSR(numVals);

		for(int h = 0; h < thisColsSize; h++) {
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);

			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;
			int retIdx = 0;
			for(int i = sPos; i < sEnd; i++) {
				while(retIdx < aggColSize && aggregateColumns.get(retIdx) < sIndexes[i])
					retIdx++;

				if(retIdx == aggColSize)
					break;
				ret.add(h, retIdx, sValues[i]);
			}

		}

		final MatrixBlock retB = new MatrixBlock(numVals, aggregateColumns.size(), -1, ret);
		retB.recomputeNonZeros();
		return MatrixBlockDictionary.create(retB, false);
	}

	@Override
	public IDictionary append(double[] row) {
		return getMBDict().append(row);
	}

	@Override
	public String getString(int colIndexes) {
		return "IdentityMatrix of size: " + nRowCol + " with empty: " + withEmpty;
	}

	@Override
	public String toString() {
		return "IdentityMatrix of size: " + nRowCol + " with empty: " + withEmpty;
	}

}
