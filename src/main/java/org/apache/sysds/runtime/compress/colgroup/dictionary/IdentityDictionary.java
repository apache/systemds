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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

/**
 * A specialized dictionary that exploits the fact that the contained dictionary is an Identity Matrix.
 */
public class IdentityDictionary extends AIdentityDictionary {

	private static final long serialVersionUID = 2535887782153955098L;

	/**
	 * Create an identity matrix dictionary. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol The number of rows and columns in this identity matrix.
	 */
	private IdentityDictionary(int nRowCol) {
		super(nRowCol);
	}

	/**
	 * Create an identity matrix dictionary. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol The number of rows and columns in this identity matrix.
	 * @return a Dictionary instance.
	 */
	public static IDictionary create(int nRowCol) {
		return create(nRowCol, false);
	}

	/**
	 * Create an identity matrix dictionary, It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol   The number of rows and columns in this identity matrix.
	 * @param withEmpty If the matrix should contain an empty row in the end.
	 */
	private IdentityDictionary(int nRowCol, boolean withEmpty) {
		super(nRowCol, withEmpty);
	}

	/**
	 * Create an identity matrix dictionary, It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol   The number of rows and columns in this identity matrix.
	 * @param withEmpty If the matrix should contain an empty row in the end.
	 * @return a Dictionary instance.
	 */
	public static IDictionary create(int nRowCol, boolean withEmpty) {
		if(nRowCol == 1) {
			if(withEmpty)
				return new Dictionary(new double[] {1, 0});
			else
				return new Dictionary(new double[] {1});
		}
		return new IdentityDictionary(nRowCol, withEmpty);
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

	@Override
	public double getValue(int r, int c, int nCol) {
		return r == c ? 1 : 0;
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(-1); // int + padding + softReference
	}

	public static long getInMemorySize(int numberColumns) {
		return AIdentityDictionary.getInMemorySize(numberColumns);
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
	public double[] aggregateRows(Builtin fn, int nCol) {
		double[] ret = new double[nRowCol];
		Arrays.fill(ret, fn.execute(1, 0));
		return ret;
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
	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		if(idxStart == 0 && idxEnd == nRowCol)
			return new IdentityDictionary(nRowCol, withEmpty);
		else
			return IdentityDictionarySlice.create(nRowCol, withEmpty, idxStart, idxEnd);
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		return (long) sum(counts, nCol);
	}

	@Override
	public int[] countNNZZeroColumns(int[] counts) {
		if(withEmpty)
			return Arrays.copyOf(counts, nRowCol); // one less.
		return counts; // interesting ... but true.
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
	public MatrixBlockDictionary getMBDict() {
		return getMBDict(nRowCol);
	}

	@Override
	public MatrixBlockDictionary createMBDict(int nCol) {
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
	public boolean equals(IDictionary o) {
		if(o instanceof IdentityDictionary && //
			((IdentityDictionary) o).nRowCol == nRowCol && //
			((IdentityDictionary) o).withEmpty == withEmpty)
			return true;
		return getMBDict().equals(o);
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
	public String getString(int colIndexes) {
		return "IdentityMatrix of size: " + nRowCol + " with empty: " + withEmpty;
	}

	@Override
	public String toString() {
		return "IdentityMatrix of size: " + nRowCol + " with empty: " + withEmpty;
	}

}
