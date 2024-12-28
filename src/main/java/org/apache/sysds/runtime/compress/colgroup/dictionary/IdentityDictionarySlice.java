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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class IdentityDictionarySlice extends IdentityDictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	/** Lower index for the slice */
	private final int l;
	/** Upper index for the slice (not inclusive) */
	private final int u;

	/**
	 * Create a Identity matrix dictionary slice. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol   the number of rows and columns in this identity matrix.
	 * @param withEmpty If the matrix should contain an empty row in the end.
	 * @param l         the index lower to start at
	 * @param u         the index upper to end at (not inclusive)
	 */
	public IdentityDictionarySlice(int nRowCol, boolean withEmpty, int l, int u) {
		super(nRowCol, withEmpty);
		if(u > nRowCol || l < 0 || l >= u)
			throw new DMLRuntimeException("Invalid slice Identity: " + nRowCol + " range: " + l + "--" + u);
		this.l = l;
		this.u = u;
	}

	@Override
	public double[] getValues() {
		LOG.warn("Should not call getValues on Identity Dictionary");
		int nCol = u - l;
		double[] ret = new double[nCol * (nRowCol + (withEmpty ? 1 : 0))];
		for(int i = l; i < u; i++) {
			ret[(i * nCol) + (i - l)] = 1;
		}
		return ret;
	}

	@Override
	public double getValue(int i) {
		throw new NotImplementedException();
	}

	@Override
	public final double getValue(int r, int c, int nCol) {
		if(r < l || r > u)
			return 0;
		return super.getValue(r - l, c, nCol);
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
	public IDictionary clone() {
		return new IdentityDictionarySlice(nRowCol, withEmpty, l, u);
	}

	@Override
	public DictType getDictType() {
		return DictType.IdentitySlice;
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
		throw new NotImplementedException();
		// for(int i = 0; i < colIndexes.size(); i++) {
		// 	res[colIndexes.get(i)] = 0;
		// }
	}

	@Override
	public double sum(int[] counts, int ncol) {
		throw new NotImplementedException();
		// // number of rows, change this.
		// double s = 0.0;
		// for(int i = l; i < u; i++)
		// 	s += counts[i];
		// return s;
	}

	@Override
	public double sumSq(int[] counts, int ncol) {
		return sum(counts, ncol);
	}

	@Override
	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		return getMBDict().sliceOutColumnRange(idxStart, idxEnd, previousNumberOfColumns);
	}

	@Override
	public boolean containsValue(double pattern) {
		return pattern == 0.0 || pattern == 1.0;
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		return (long) sum(counts, nCol);
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return nRowCol + (withEmpty ? 1 : 0);
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
			identity.set(i, i - l, 1.0);
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
	public IDictionary scaleTuples(int[] scaling, int nCol) {
		return getMBDict().scaleTuples(scaling, nCol);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.IDENTITY_SLICE.ordinal());
		out.writeInt(nRowCol);
		out.writeBoolean(withEmpty);
		out.writeInt(l);
		out.writeInt(u);
	}

	public static IdentityDictionarySlice read(DataInput in) throws IOException {

		int nRowCol = in.readInt();
		boolean empty = in.readBoolean();
		int l = in.readInt();
		int u = in.readInt();

		return new IdentityDictionarySlice(nRowCol, empty, l, u);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 * 3 + 1;
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
	public double getSparsity() {
		return (double) (u - l) / ((u -l) * (nRowCol + (withEmpty ? 1 : 0)));
	}

	@Override
	public IDictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		return getMBDict().binOpRight(op, v);
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
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol, int rep) {
		if(fr >= l && fr < u) 
			v[to * nCol + fr - l] += rep;
	}

	@Override
	public boolean equals(IDictionary o) {
		if(o instanceof IdentityDictionarySlice) {
			IdentityDictionarySlice os = ((IdentityDictionarySlice) o);
			return os.nRowCol == nRowCol && os.l == l && os.u == u;
		}
		else if(o instanceof IdentityDictionary)
			return false;
		else
			return getMBDict().equals(o);
	}

	@Override
	public MatrixBlockDictionary getMBDict() {
		final int nCol = u - l;
		MatrixBlock mb = new MatrixBlock(nRowCol + (withEmpty ? 1 : 0), nCol, true);
		mb.allocateSparseRowsBlock();

		SparseBlock sb = mb.getSparseBlock();
		for(int i = l; i < u; i++) {
			sb.append(i, i - l, 1);
		}

		mb.setNonZeros(nCol);
		return new MatrixBlockDictionary(mb);
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		getMBDict().multiplyScalar(v, ret, off, dictIdx, cols);
	}

}
