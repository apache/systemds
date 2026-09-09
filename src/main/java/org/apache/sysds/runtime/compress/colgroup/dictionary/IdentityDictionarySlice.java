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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class IdentityDictionarySlice extends AIdentityDictionary {

	private static final long serialVersionUID = 2535887782153555098L;

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
		this.l = l;
		this.u = u;
	}

	/**
	 * Create a Identity matrix dictionary slice (if other groups are not more applicable). It behaves as if allocated a
	 * Sparse Matrix block but exploits that the structure is known to have certain properties.
	 * 
	 * @param nRowCol   the number of rows and columns in this identity matrix.
	 * @param withEmpty If the matrix should contain an empty row in the end.
	 * @param l         the index lower to start at
	 * @param u         the index upper to end at (not inclusive)
	 * @return a Dictionary instance.
	 */
	public static IDictionary create(int nRowCol, boolean withEmpty, int l, int u) {
		if(u > nRowCol || l < 0 || l >= u)
			throw new DMLRuntimeException("Invalid slice Identity: " + nRowCol + " range: " + l + "--" + u);
		if(nRowCol == 1) {
			if(withEmpty)
				return new Dictionary(new double[] {1, 0});
			else
				return new Dictionary(new double[] {1});
		}
		else if(l == 0 && u == nRowCol)
			return IdentityDictionary.create(nRowCol, withEmpty);
		else
			return new IdentityDictionarySlice(nRowCol, withEmpty, l, u);
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
		final int nCol = u - l;
		final int vRow = i / nCol;
		if(vRow < l || vRow >= u)
			return 0;
		final int oRow = vRow - l;
		final int col = i % nCol;
		return oRow == col ? 1 : 0;
	}

	@Override
	public final double getValue(int r, int c, int nCol) {
		if(r < l || r > u)
			return 0;
		return (r - l) == c ? 1 : 0;
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(nRowCol);
	}

	public static long getInMemorySize(int numberColumns) {
		// 2 more ints, no padding.
		return AIdentityDictionary.getInMemorySize(numberColumns) + 8;
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0)];
		if(l + 1 == u) {
			ret[l] = 1;
			return ret;
		}
		else {
			Arrays.fill(ret, l, u, fn.execute(1, 0));
			return ret;
		}
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
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0)];
		Arrays.fill(ret, l, u, 1.0);
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[getNumberOfValues(defaultTuple.length) + 1];
		for(int i = l; i < u; i++)
			ret[i] = 1;
		for(int i = 0; i < defaultTuple.length; i++)
			ret[ret.length - 1] += defaultTuple[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		final double[] ret = new double[getNumberOfValues(reference.length)];
		double refSum = 0;
		for(int i = 0; i < reference.length; i++)
			refSum += reference[i];
		for(int i = 0; i < l; i++)
			ret[i] = refSum;
		for(int i = l; i < u; i++)
			ret[i] = 1 + refSum;
		for(int i = u; i < ret.length; i++)
			ret[i] = refSum;
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0)];
		Arrays.fill(ret, l, u, 1);
		return ret;
	}

	@Override
	public double[] productAllRowsToDouble(int nCol) {
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0)];
		if(u - l - 1 == 0)
			ret[l] = 1;
		return ret;
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		int nVal = nRowCol + (withEmpty ? 1 : 0);
		double[] ret = new double[nVal + 1];
		if(u - l - 1 == 0)
			ret[l] = 1;
		ret[nVal] = defaultTuple[0];
		for(int i = 1; i < defaultTuple.length; i++)
			ret[nVal] *= defaultTuple[i];
		return ret;
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		for(int i = l; i < u; i++)
			c[colIndexes.get(i - l)] = counts[i];
	}

	@Override
	public double sum(int[] counts, int ncol) {
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
	public long getNumberNonZeros(int[] counts, int nCol) {
		return (long) sum(counts, nCol);
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return nRowCol + (withEmpty ? 1 : 0);
	}

	@Override
	public int getNumberOfColumns(int nrow) {
		if(nrow != (nRowCol + (withEmpty ? 1 : 0)))
			throw new DMLCompressionException("Invalid call to get Number of values assuming wrong number of columns");
		return u - l;
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
	public double getSparsity() {
		return (double) (u - l) / ((u - l) * (nRowCol + (withEmpty ? 1 : 0)));
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
			return os.nRowCol == nRowCol && os.l == l && os.u == u && withEmpty == os.withEmpty;
		}
		else if(o instanceof IdentityDictionary)
			return false;
		else
			return getMBDict().equals(o);
	}

	@Override
	public MatrixBlockDictionary getMBDict() {
		return getMBDict(nRowCol);
	}

	@Override
	public MatrixBlockDictionary createMBDict(int nCol) {
		MatrixBlock identity = new MatrixBlock(nRowCol + (withEmpty ? 1 : 0), u - l, true);
		for(int i = l; i < u; i++)
			identity.set(i, i - l, 1.0);
		return new MatrixBlockDictionary(identity);
	}

	@Override
	public String getString(int colIndexes) {
		return toString();
	}

	@Override 
	public IDictionary sliceColumns(IntArrayList selectedColumns, int nCol){
		return getMBDict().sliceColumns(selectedColumns, nCol);
	}

	@Override
	public String toString() {
		return "IdentityMatrixSlice of size: " + nRowCol + " l " + l + " u " + u;
	}

}
