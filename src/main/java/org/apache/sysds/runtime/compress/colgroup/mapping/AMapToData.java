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

package org.apache.sysds.runtime.compress.colgroup.mapping;

import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * This Class's job is to link into the dictionary entries for column groups.
 * 
 * Column groups
 * 
 * - DDC use this to map to map directly to the dictionary
 * 
 * - SDC use this in collaboration with the offsets to only point to dictionary entries for non default values.
 */
public abstract class AMapToData implements Serializable {
	private static final long serialVersionUID = 1208906071822976041L;
	protected static final Log LOG = LogFactory.getLog(AMapToData.class.getName());

	/** Number of unique values inside this map. */
	private int nUnique;

	/**
	 * Main constructor for AMapToData.
	 * 
	 * NOTE! The value should be representable inside the map. This requirement is not checked.
	 * 
	 * @param nUnique number of unique values.
	 */
	protected AMapToData(int nUnique) {
		this.nUnique = nUnique;
	}

	/**
	 * Get the number of unique values inside this map.
	 * 
	 * @return the unique count.
	 */
	public final int getUnique() {
		return nUnique;
	}

	/**
	 * Set number of unique values.
	 * 
	 * NOTE! The value should be representable inside the map. This requirement is not checked.
	 * 
	 * @param nUnique the value to set.
	 */
	public final void setUnique(int nUnique) {
		this.nUnique = nUnique;
	}

	/**
	 * Get the given index back as a integer
	 * 
	 * @param n the index to get
	 * @return the value represented in that cell as integer
	 */
	public abstract int getIndex(int n);

	/**
	 * Shortcut method to support Integer objects, not really efficient but for the purpose of reusing code.
	 * 
	 * @param n The index to set.
	 * @param v The value to set.
	 */
	public void set(int n, Integer v) {
		set(n, (int) v);
	}

	/**
	 * Set the index to the value.
	 * 
	 * NOTE! The value should be representable inside the map. This requirement is not checked.
	 * 
	 * @param n index to set.
	 * @param v the value to set it to.
	 */
	public abstract void set(int n, int v);

	/**
	 * set a range of values from another map.
	 * 
	 * The given tm must only contain supported values, and it is not verified.
	 * 
	 * @param l   lower bound
	 * @param u   upper bound (not inclusive)
	 * @param off offset to take values from tm
	 * @param tm  the other map to copy values from
	 */
	public abstract void set(int l, int u, int off, AMapToData tm);

	/**
	 * Set the index to the value and get the contained value after.
	 * 
	 * @param n index to set.
	 * @param v the value to set it to.
	 * @return v as encoded, note this value can be different that the one put in if the map is not able to represent the
	 *         value
	 */
	public abstract int setAndGet(int n, int v);

	/**
	 * Fill the map with a given value.
	 * 
	 * NOTE! The value should be representable inside the map. This requirement is not checked.
	 * 
	 * @param v the value to fill
	 */
	public abstract void fill(int v);

	/**
	 * Get the maximum value that is possible to allocate inside this map.
	 * 
	 * @return The maximum value.
	 */
	public abstract int getUpperBoundValue();

	/**
	 * Get the in memory size of this Mapping object.
	 * 
	 * @return The size in Bytes.
	 */
	public abstract long getInMemorySize();

	/**
	 * Get the size of this Mapping object on disk.
	 * 
	 * @return The on disk size in Bytes.
	 */
	public abstract long getExactSizeOnDisk();

	/**
	 * The size of the Mapping object, signaling how many value cells are stored in this mapping object.
	 * 
	 * @return The length of the mapping object.
	 */
	public abstract int size();

	/**
	 * Serialize this object to the DataOutput given.
	 * 
	 * @param out The object to serialize this object into.
	 * @throws IOException An IO exception if the Serialization fails.
	 */
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Replace v with r for all entries,
	 * 
	 * NOTE! It is assumed that you call this correctly:
	 * 
	 * - with two distinct values that is representable inside the given AMapToData.
	 * 
	 * @param v The value to replace
	 * @param r The value to put instead
	 */
	public abstract void replace(int v, int r);

	public abstract MAP_TYPE getType();

	/**
	 * Pre aggregate a dense matrix m into pre, subject to only including a row segment and column segment.
	 * 
	 * @param m     The dense matrix values to preaggregate
	 * @param preAV The preAggregate double array populate with the summed values of m
	 * @param rl    The row start in m
	 * @param ru    The row end in m
	 * @param cl    The column start in m
	 * @param cu    The column end in m
	 */
	public final void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu) {
		final DenseBlock db = m.getDenseBlock();
		if(rl == ru - 1)
			preAggregateDenseSingleRow(db.values(rl), db.pos(rl), preAV, cl, cu);
		else
			preAggregateDenseMultiRow(m, preAV, rl, ru, cl, cu);
	}

	/**
	 * PreAggregate Dense on a single row.
	 * 
	 * @param mV    The DenseMatrix Values from the input matrix block for the specific row given
	 * @param off   The offset into the mV that the row values start from
	 * @param preAV The PreAggregate value target to preAggregate into
	 * @param cl    The column index to start at
	 * @param cu    The column index to stop at (not inclusive)
	 */
	protected void preAggregateDenseSingleRow(double[] mV, int off, double[] preAV, int cl, int cu) {
		if(cu - cl > 64)
			preAggregateDenseToRowBy8(mV, preAV, cl, cu, off);
		else {
			off += cl;
			for(int rc = cl; rc < cu; rc++, off++)
				preAV[getIndex(rc)] += mV[off];
		}
	}

	protected void preAggregateDenseToRowBy8(double[] mV, double[] preAV, int cl, int cu, int off) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[getIndex(rc)] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8)
			preAggregateDenseToRowVec8(mV, preAV, rc, off);
	}

	protected void preAggregateDenseToRowVec8(double[] mV, double[] preAV, int rc, int off) {
		preAV[getIndex(rc)] += mV[off];
		preAV[getIndex(rc + 1)] += mV[off + 1];
		preAV[getIndex(rc + 2)] += mV[off + 2];
		preAV[getIndex(rc + 3)] += mV[off + 3];
		preAV[getIndex(rc + 4)] += mV[off + 4];
		preAV[getIndex(rc + 5)] += mV[off + 5];
		preAV[getIndex(rc + 6)] += mV[off + 6];
		preAV[getIndex(rc + 7)] += mV[off + 7];
	}

	/**
	 * PreAggregate from Dense Matrix, and handle multiple rows,
	 * 
	 * @param m     The Matrix to preAggregate.
	 * @param preAV The target dense array to preAggregate into
	 * @param rl    The row to start at
	 * @param ru    The row to end at (not inclusive)
	 * @param cl    The column to start at
	 * @param cu    The column to end at (not inclusive)
	 */
	protected void preAggregateDenseMultiRow(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu) {
		final int nVal = getUnique();
		final DenseBlock db = m.getDenseBlock();
		if(db.isContiguous()) {
			final double[] mV = m.getDenseBlockValues();
			final int nCol = m.getNumColumns();
			preAggregateDenseMultiRowContiguous(mV, nCol, nVal, preAV, rl, ru, cl, cu);
		}
		else
			throw new NotImplementedException();
	}

	protected void preAggregateDenseMultiRowContiguous(double[] mV, int nCol, int nVal, double[] preAV, int rl, int ru,
		int cl, int cu) {
		if(cu - cl > 64)
			preAggregateDenseMultiRowContiguousBy8(mV, nCol, nVal, preAV, rl, ru, cl, cu);
		else
			preAggregateDenseMultiRowContiguousBy1(mV, nCol, nVal, preAV, rl, ru, cl, cu);
	}

	protected void preAggregateDenseMultiRowContiguousBy8(double[] mV, int nCol, int nVal, double[] preAV, int rl,
		int ru, int cl, int cu) {
		final int h = (cu - cl) % 8;
		preAggregateDenseMultiRowContiguousBy1(mV, nCol, nVal, preAV, rl, ru, cl, cl + h);
		final int offR = nCol * rl;
		final int offE = nCol * ru;
		for(int c = cl + h; c < cu; c += 8) {
			final int id1 = getIndex(c), id2 = getIndex(c + 1), id3 = getIndex(c + 2), id4 = getIndex(c + 3),
				id5 = getIndex(c + 4), id6 = getIndex(c + 5), id7 = getIndex(c + 6), id8 = getIndex(c + 7);

			final int start = c + offR;
			final int end = c + offE;
			int nValOff = 0;
			for(int off = start; off < end; off += nCol) {
				preAV[id1 + nValOff] += mV[off];
				preAV[id2 + nValOff] += mV[off + 1];
				preAV[id3 + nValOff] += mV[off + 2];
				preAV[id4 + nValOff] += mV[off + 3];
				preAV[id5 + nValOff] += mV[off + 4];
				preAV[id6 + nValOff] += mV[off + 5];
				preAV[id7 + nValOff] += mV[off + 6];
				preAV[id8 + nValOff] += mV[off + 7];
				nValOff += nVal;
			}
		}
	}

	protected void preAggregateDenseMultiRowContiguousBy1(double[] mV, int nCol, int nVal, double[] preAV, int rl,
		int ru, int cl, int cu) {
		final int offR = nCol * rl;
		final int offE = nCol * ru;
		for(int c = cl; c < cu; c++) {
			final int idx = getIndex(c);
			final int start = c + offR;
			final int end = c + offE;
			for(int offOut = idx, off = start; off < end; offOut += nVal, off += nCol) {
				preAV[offOut] += mV[off];
			}
		}
	}

	/**
	 * PreAggregate a Dense Matrix at index offsets.
	 * 
	 * @param m       The DenseBlock to preAggregate
	 * @param preAV   The target double array to put the preAggregate into
	 * @param rl      The row to start at
	 * @param ru      The row to end at (not inclusive)
	 * @param cl      The column in m to start from
	 * @param cu      The column in m to end at (not inclusive)
	 * @param indexes The Offset Indexes to iterate through
	 */
	public final void preAggregateDense(DenseBlock m, double[] preAV, int rl, int ru, int cl, int cu, AOffset indexes) {
		indexes.preAggregateDenseMap(m, preAV, rl, ru, cl, cu, getUnique(), this);
	}

	/**
	 * PreAggregate the SparseBlock in the range of rows given.
	 * 
	 * @param sb      The SparseBlock to preAggregate
	 * @param preAV   The target double array to put the preAggregate into
	 * @param rl      The row to start at
	 * @param ru      The row to end at (not inclusive)
	 * @param indexes The Offset Indexes to iterate through
	 */
	public final void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes) {
		indexes.preAggSparseMap(sb, preAV, rl, ru, getUnique(), this);
	}

	/**
	 * PreAggregate the sparseblock in the range of rows given.
	 * 
	 * @param sb    Sparse block to preAggregate from
	 * @param preAV Pre aggregate target
	 * @param rl    row index in sb
	 * @param ru    upper row index in sp (not inclusive)
	 */
	public final void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru) {
		if(rl == ru - 1)
			preAggregateSparseSingleRow(sb, preAV, rl);
		else
			preAggregateSparseMultiRow(sb, preAV, rl, ru);
	}

	private final void preAggregateSparseSingleRow(final SparseBlock sb, final double[] preAV, final int r) {
		if(sb.isEmpty(r))
			return;
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);
		for(int j = apos; j < alen; j++)
			preAV[getIndex(aix[j])] += avals[j];
	}

	private final void preAggregateSparseMultiRow(final SparseBlock sb, final double[] preAV, final int rl,
		final int ru) {
		final int unique = getUnique();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final int[] aix = sb.indexes(r);
			final double[] avals = sb.values(r);
			final int off = unique * (r - rl);
			for(int j = apos; j < alen; j++)
				preAV[off + getIndex(aix[j])] += avals[j];
		}
	}

	/**
	 * Get the number of counts of each unique value contained in this map. Note that in the case the mapping is shorter
	 * than number of rows the counts sum to the number of mapped values not the number of rows.
	 * 
	 * @return The counts
	 */
	public final int[] getCounts() {
		return getCounts(new int[getUnique()]);
	}

	/**
	 * Get the number of counts of each unique value contained in this map. Note that in the case the mapping is shorter
	 * than number of rows the counts sum to the number of mapped values not the number of rows.
	 * 
	 * @param counts The object to return.
	 * @return The counts
	 */
	public abstract int[] getCounts(int[] counts);

	/**
	 * PreAggregate into dictionary with two sides of DDC.
	 * 
	 * @param tm   Map of other side
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns
	 */
	public final void preAggregateDDC_DDC(AMapToData tm, IDictionary td, Dictionary ret, int nCol) {
		if(nCol == 1)
			preAggregateDDC_DDCSingleCol(tm, td.getValues(), ret.getValues());
		else
			preAggregateDDC_DDCMultiCol(tm, td, ret.getValues(), nCol);
	}

	/**
	 * PreAggregate into dictionary with two sides of DDC guaranteed to only have one column tuples.
	 * 
	 * @param tm  Map of other side
	 * @param td  Dictionary to take values from (other side dictionary)
	 * @param ret The output dictionary to aggregate into
	 */
	protected void preAggregateDDC_DDCSingleCol(AMapToData tm, double[] td, double[] v) {

		final int sz = size();
		final int h = sz % 8;
		for(int r = 0; r < h; r++)
			v[getIndex(r)] += td[tm.getIndex(r)];
		for(int r = h; r < sz; r += 8)
			preAggregateDDC_DDCSingleCol_vec(tm, td, v, r);

	}

	protected void preAggregateDDC_DDCSingleCol_vec(AMapToData tm, double[] td, double[] v, int r) {
		final int r2 = r + 1, r3 = r + 2, r4 = r + 3, r5 = r + 4, r6 = r + 5, r7 = r + 6, r8 = r + 7;
		v[getIndex(r)] += td[tm.getIndex(r)];
		v[getIndex(r2)] += td[tm.getIndex(r2)];
		v[getIndex(r3)] += td[tm.getIndex(r3)];
		v[getIndex(r4)] += td[tm.getIndex(r4)];
		v[getIndex(r5)] += td[tm.getIndex(r5)];
		v[getIndex(r6)] += td[tm.getIndex(r6)];
		v[getIndex(r7)] += td[tm.getIndex(r7)];
		v[getIndex(r8)] += td[tm.getIndex(r8)];
	}

	/**
	 * PreAggregate into dictionary with two sides of DDC guaranteed to multiple column tuples.
	 * 
	 * @param tm   Map of other side that indicate the indexes to take out and put into ret
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns in td
	 */
	protected void preAggregateDDC_DDCMultiCol(final AMapToData tm, final IDictionary td, final double[] v,
		final int nCol) {

		final int sz = size();
		final int h = sz % 8;
		for(int r = 0; r < h; r++)
			td.addToEntry(v, tm.getIndex(r), getIndex(r), nCol);
		for(int r = h; r < sz; r += 8)
			preAggregateDDC_DDCMultiCol_vec(tm, td, v, nCol, r);

	}

	protected void preAggregateDDC_DDCMultiCol_vec(final AMapToData tm, final IDictionary td, final double[] v,
		final int nCol, final int r) {
		final int r2 = r + 1, r3 = r + 2, r4 = r + 3, r5 = r + 4, r6 = r + 5, r7 = r + 6, r8 = r + 7;
		td.addToEntryVectorized(v, //
			tm.getIndex(r), tm.getIndex(r2), tm.getIndex(r3), tm.getIndex(r4), tm.getIndex(r5), tm.getIndex(r6),
			tm.getIndex(r7), tm.getIndex(r8), getIndex(r), //
			getIndex(r2), getIndex(r3), getIndex(r4), getIndex(r5), getIndex(r6), getIndex(r7), getIndex(r8), nCol);
	}

	/**
	 * PreAggregate into SDCZero dictionary from DDC dictionary.
	 * 
	 * @param tm   Map of other side
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param tof  The offset index structure of the SDC side
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns in output and td dictionary
	 */
	public final void preAggregateDDC_SDCZ(AMapToData tm, IDictionary td, AOffset tof, Dictionary ret, int nCol) {
		if(nCol == 1)
			preAggregateDDC_SDCZSingleCol(tm, td.getValues(), tof, ret.getValues());
		else
			preAggregateDDC_SDCZMultiCol(tm, td, tof, ret.getValues(), nCol);
	}

	public void preAggregateDDC_SDCZSingleCol(AMapToData tm, double[] td, AOffset tof, double[] v) {
		final AOffsetIterator itThat = tof.getOffsetIterator();
		final int size = tm.size() - 1;
		for(int i = 0; i < size; i++) {
			final int to = getIndex(itThat.value());
			final int fr = tm.getIndex(i);
			v[to] += td[fr];
			itThat.next();
		}
		final int to = getIndex(itThat.value());
		final int fr = tm.getIndex(size);
		v[to] += td[fr];
	}

	public void preAggregateDDC_SDCZMultiCol(AMapToData tm, IDictionary td, AOffset tof, double[] v, int nCol) {
		final AOffsetIterator it = tof.getOffsetIterator();
		final int size = tm.size() - 1;
		int i = (size > 8) ? preAggregateDDC_SDCZMultiCol_vect(tm, td, v, nCol, it, size) : 0;

		for(; i < size; i++) {
			final int to = getIndex(it.value());
			final int fr = tm.getIndex(i);
			td.addToEntry(v, fr, to, nCol);
			it.next();
		}

		final int to = getIndex(it.value());
		final int fr = tm.getIndex(size);
		td.addToEntry(v, fr, to, nCol);
	}

	private int preAggregateDDC_SDCZMultiCol_vect(AMapToData tm, IDictionary td, double[] v, int nCol,
		AOffsetIterator it, int size) {
		final int h = size % 8;
		int i = 0;
		while(i < size - h) {
			int t1 = it.value(), t2 = it.next(), t3 = it.next(), t4 = it.next(), t5 = it.next(), t6 = it.next(),
				t7 = it.next(), t8 = it.next();

			t1 = getIndex(t1);
			t2 = getIndex(t2);
			t3 = getIndex(t3);
			t4 = getIndex(t4);
			t5 = getIndex(t5);
			t6 = getIndex(t6);
			t7 = getIndex(t7);
			t8 = getIndex(t8);

			int f1 = tm.getIndex(i), f2 = tm.getIndex(i + 1), f3 = tm.getIndex(i + 2), f4 = tm.getIndex(i + 3),
				f5 = tm.getIndex(i + 4), f6 = tm.getIndex(i + 5), f7 = tm.getIndex(i + 6), f8 = tm.getIndex(i + 7);

			i += 8;
			it.next();
			td.addToEntryVectorized(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
		}
		return i;
	}

	/**
	 * PreAggregate into DDC dictionary from SDCZero dictionary.
	 * 
	 * @param tm   Map of other side
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param of   Offsets of the SDC to look into DDC
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns in output and td dictionary
	 */
	public final void preAggregateSDCZ_DDC(AMapToData tm, IDictionary td, AOffset of, Dictionary ret, int nCol) {
		if(nCol == 1)
			preAggregateSDCZ_DDCSingleCol(tm, td.getValues(), of, ret.getValues());
		else
			preAggregateSDCZ_DDCMultiCol(tm, td, of, ret.getValues(), nCol);
	}

	protected void preAggregateSDCZ_DDCSingleCol(AMapToData tm, double[] td, AOffset of, double[] v) {
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int size = size() - 1;
		int tv = itThis.value();
		for(int i = 0; i < size; i++) {
			v[getIndex(i)] += td[tm.getIndex(tv)];
			tv = itThis.next();
		}
		v[getIndex(size)] += td[tm.getIndex(tv)];
	}

	protected void preAggregateSDCZ_DDCMultiCol(AMapToData tm, IDictionary td, AOffset of, double[] v, int nCol) {
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int size = size() - 1;
		int i = (size > 8) ? preAggregateSDCZ_DDCMultiCol_vect(tm, td, v, nCol, itThis, size) : 0;

		int tv = itThis.value();
		for(; i < size; i++) {
			td.addToEntry(v, tm.getIndex(tv), getIndex(i), nCol);
			tv = itThis.next();
		}
		td.addToEntry(v, tm.getIndex(tv), getIndex(size), nCol);
	}

	private int preAggregateSDCZ_DDCMultiCol_vect(AMapToData tm, IDictionary td, double[] v, int nCol,
		AOffsetIterator it, int size) {
		final int h = size % 8;
		int i = 0;
		while(i < size - h) {
			int t1 = getIndex(i), t2 = getIndex(i + 1), t3 = getIndex(i + 2), t4 = getIndex(i + 3), t5 = getIndex(i + 4),
				t6 = getIndex(i + 5), t7 = getIndex(i + 6), t8 = getIndex(i + 7);

			int f1 = it.value(), f2 = it.next(), f3 = it.next(), f4 = it.next(), f5 = it.next(), f6 = it.next(),
				f7 = it.next(), f8 = it.next();

			f1 = tm.getIndex(f1);
			f2 = tm.getIndex(f2);
			f3 = tm.getIndex(f3);
			f4 = tm.getIndex(f4);
			f5 = tm.getIndex(f5);
			f6 = tm.getIndex(f6);
			f7 = tm.getIndex(f7);
			f8 = tm.getIndex(f8);

			i += 8;
			it.next();
			td.addToEntryVectorized(v, f1, f2, f3, f4, f5, f6, f7, f8, t1, t2, t3, t4, t5, t6, t7, t8, nCol);
		}
		return i;
	}

	public final void preAggregateSDCZ_SDCZ(AMapToData tm, IDictionary td, AOffset tof, AOffset of, Dictionary ret,
		int nCol) {
		if(nCol == 1)
			preAggregateSDCZ_SDCZSingleCol(tm, td.getValues(), tof, of, ret.getValues());
		else
			preAggregateSDCZ_SDCZMultiCol(tm, td, tof, of, ret.getValues(), nCol);
	}

	private final void preAggregateSDCZ_SDCZSingleCol(AMapToData tm, double[] td, AOffset tof, AOffset of, double[] dv) {
		final AOffsetIterator itThat = tof.getOffsetIterator();
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int tSize = tm.size() - 1, size = size() - 1;
		preAggregateSDCZ_SDCZSingleCol(tm, td, dv, itThat, itThis, tSize, size);
	}

	protected void preAggregateSDCZ_SDCZSingleCol(AMapToData tm, double[] td, double[] dv, AOffsetIterator itThat,
		AOffsetIterator itThis, int tSize, int size) {

		int i = 0, j = 0, tv = itThat.value(), v = itThis.value();

		// main preAggregate process
		while(i < tSize && j < size) {
			if(tv == v) {
				dv[getIndex(j)] += td[tm.getIndex(i)];
				tv = itThat.next();
				v = itThis.next();
				i++;
				j++;
			}
			else if(tv < v) {
				tv = itThat.next();
				i++;
			}
			else {
				v = itThis.next();
				j++;
			}
		}

		// Remaining part (very small so not really main performance bottleneck)
		preAggregateSDCZ_SDCZMultiCol_tail(tm, this, Dictionary.create(td), dv, 1, itThat, itThis, tSize, size, i, j);
	}

	protected void preAggregateSDCZ_SDCZMultiCol(AMapToData tm, IDictionary td, AOffset tof, AOffset of, double[] dv,
		int nCol) {
		final AOffsetIterator itThat = tof.getOffsetIterator();
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int tSize = tm.size() - 1, size = size() - 1;
		int i = 0, j = 0;

		// main preAggregate process
		while(i < tSize && j < size) {
			final int tv = itThat.value();
			final int v = itThis.value();
			if(tv == v) {
				final int fr = tm.getIndex(i);
				final int to = getIndex(j);
				td.addToEntry(dv, fr, to, nCol);
				itThat.next();
				itThis.next();
				i++;
				j++;
			}
			else if(tv < v) {
				itThat.next();
				i++;
			}
			else {
				itThis.next();
				j++;
			}
		}

		// Remaining part (very small so not really main performance bottleneck)
		preAggregateSDCZ_SDCZMultiCol_tail(tm, this, td, dv, nCol, itThat, itThis, tSize, size, i, j);
	}

	protected static void preAggregateSDCZ_SDCZMultiCol_tail(AMapToData tm, AMapToData m, IDictionary td, double[] dv,
		int nCol, AOffsetIterator itThat, AOffsetIterator itThis, int tSize, int size, int i, int j) {
		int tv = itThat.value();
		int v = itThis.value();
		if(tv == v) {
			final int fr = tm.getIndex(i);
			final int to = m.getIndex(j);
			td.addToEntry(dv, fr, to, nCol);
			return;
		}

		while(i < tSize && tv < v) { // this is at final
			itThat.next();
			i++;
			tv = itThat.value();
			if(tv == v) {
				final int fr = tm.getIndex(i);
				final int to = m.getIndex(j);
				td.addToEntry(dv, fr, to, nCol);
				return;
			}
		}

		while(j < size && v < tv) { // that is at final
			itThis.next();
			j++;
			v = itThis.value();
			if(tv == v) {
				final int fr = tm.getIndex(i);
				final int to = m.getIndex(j);
				td.addToEntry(dv, fr, to, nCol);
				return;
			}
		}
	}

	public void preAggregateRLE_DDC(int[] ptr, char[] data, IDictionary td, Dictionary ret, int nCol) {
		if(nCol == 1)
			preAggregateRLE_DDCSingleCol(ptr, data, td.getValues(), ret.getValues());
		else
			preAggregateRLE_DDCMultiCol(ptr, data, td, ret.getValues(), nCol);
	}

	protected void preAggregateRLE_DDCSingleCol(int[] ptr, char[] data, double[] td, double[] ret) {
		// find each index in RLE, and aggregate into those.
		for(int k = 0; k < ret.length; k++) { // for each run in RLE
			final int blen = ptr[k + 1];
			for(int apos = ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + data[apos];
				re = rs + data[apos + 1];
				for(int rix = rs; rix < re; rix++)
					ret[k] += td[getIndex(rix)];
			}
		}
	}

	protected void preAggregateRLE_DDCMultiCol(int[] ptr, char[] data, IDictionary td, double[] ret, int nCol) {
		// find each index in RLE, and aggregate into those.
		for(int k = 0; k < ret.length / nCol; k++) { // for each run in RLE
			final int blen = ptr[k + 1];
			for(int apos = ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + data[apos];
				re = rs + data[apos + 1];
				for(int rix = rs; rix < re; rix++)
					td.addToEntry(ret, getIndex(rix), k, nCol);
			}
		}
	}

	public void preAggregateDDC_RLE(int[] ptr, char[] data, IDictionary td, Dictionary ret, int nCol) {
		// find each index in RLE, and aggregate into those.
		double[] v = ret.getValues();
		for(int k = 0; k < ptr.length - 1; k++) { // for each run in RLE
			final int blen = ptr[k + 1];
			for(int apos = ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + data[apos];
				re = rs + data[apos + 1];
				for(int rix = rs; rix < re; rix++)
					td.addToEntry(v, k, getIndex(rix), nCol);
			}
		}
	}

	/**
	 * Copy the values in this map into another mapping object.
	 * 
	 * NOTE! All contained vales should be representable inside the map given. This requirement is not checked.
	 * 
	 * @param d Map to copy all values into.
	 */
	public void copy(AMapToData d) {
		if(d.nUnique == 1)
			fill(0);
		else if(d instanceof MapToBit)
			copyBit((MapToBit) d);
		else if(d instanceof MapToInt)
			copyInt((MapToInt) d);
		else {
			final int sz = size();
			for(int i = 0; i < sz; i++)
				set(i, d.getIndex(i));
		}
	}

	protected void copyInt(MapToInt d) {
		copyInt(d.getData());
	}

	/**
	 * Copy the values of the given array into this.
	 * 
	 * Note that this operation stops at the length of this AMapToData
	 * 
	 * Therefore the given d length can not be longer than this size.
	 * 
	 * @param d The array to copy
	 */
	public void copyInt(int[] d) {
		copyInt(d, 0, Math.min(d.length, size()));
	}

	public abstract void copyInt(int[] d, int start, int end);

	public void copyBit(MapToBit d) {
		fill(0);
		for(int i = d.nextSetBit(0); i >= 0; i = d.nextSetBit(i + 1)) {
			set(i, 1);
		}
	}

	public int getMax() {
		int m = -1;
		for(int i = 0; i < size(); i++) {
			int v = getIndex(i);
			m = v > m ? v : m;
		}
		return m;
	}

	/**
	 * Reallocate the map, to a smaller instance if applicable. Note it does not change the length of the array, just the
	 * datatype.
	 * 
	 * Note that it returns the input if the input is the smallest representation that fits, otherwise it will return
	 * something that is smaller.
	 * 
	 * @param unique The number of tuples that should be supported in the resulting map
	 * @return The returned hopefully reduced map.
	 */
	public abstract AMapToData resize(int unique);

	/**
	 * Count the number of runs inside the map.
	 * 
	 * @return The number of runs
	 */
	public abstract int countRuns();

	/**
	 * Count the number of runs inside the map, but sparse with offsets.
	 * 
	 * @param off The sparse offsets to consider counting the runs from.
	 * @return count of runs.
	 */
	public int countRuns(AOffset off) {
		int c = 1;
		final int size = size();
		final AOffsetIterator of = off.getOffsetIterator();
		for(int i = 1; i < size; i++) {
			int id = of.value();
			if(id + 1 == of.next())
				c += getIndex(i - 1) == getIndex(i) ? 0 : 1;
			else
				c++;
		}
		return c;
	}

	/**
	 * Slice out the range from lower to upper from this map toData.
	 * 
	 * @param l Low value to slice from
	 * @param u high value to slice to (not inclusive)
	 * @return A new map containing only the values from the range.
	 */
	public abstract AMapToData slice(int l, int u);

	public abstract AMapToData append(AMapToData t);

	public abstract AMapToData appendN(IMapToDataGroup[] d);

	@Override
	public boolean equals(Object e) {
		return this == e || // same object or
			(e instanceof AMapToData && this.equals((AMapToData) e));
	}

	/**
	 * Indicate if the given encoding is equivalent to this encoding
	 * 
	 * @param e The other encoding to be compared with this
	 * @return If the encoding is equivalent
	 */
	public abstract boolean equals(AMapToData e);

	/** Debugging verification that this mapping is correctly made. */
	public void verify() {
		if(CompressedMatrixBlock.debug) {
			for(int i = 0; i < size(); i++) {
				if(getIndex(i) >= nUnique) {
					throw new DMLCompressionException("Invalid construction of Mapping data containing values above unique");
				}
			}
		}
	}

	public void lmSparseMatrixRow(SparseBlock sb, final int r, DenseBlock db, final IColIndex colIndexes,
		final IDictionary dict) {
		if(sb.isEmpty(r))
			return;
		// dense output blocks locations
		final int pos = db.pos(r);
		final double[] retV = db.values(r);

		// sparse left block locations
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] aval = sb.values(r);

		for(int i = apos; i < alen; i++)
			dict.multiplyScalar(aval[i], retV, pos, getIndex(aix[i]), colIndexes);
	}

	public void decompressToRange(double[] c, int rl, int ru, int offR, double[] values) {
		if(offR == 0)
			decompressToRangeNoOff(c, rl, ru, values);
		else
			decompressToRangeOff(c, rl, ru, offR, values);
	}

	protected void decompressToRangeOff(double[] c, int rl, int ru, int offR, double[] values) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++)
			c[offT] += values[getIndex(i)];
	}

	protected void decompressToRangeNoOffBy8(double[] c, int r, double[] values) {
		c[r] += values[getIndex(r)];
		c[r + 1] += values[getIndex(r + 1)];
		c[r + 2] += values[getIndex(r + 2)];
		c[r + 3] += values[getIndex(r + 3)];
		c[r + 4] += values[getIndex(r + 4)];
		c[r + 5] += values[getIndex(r + 5)];
		c[r + 6] += values[getIndex(r + 6)];
		c[r + 7] += values[getIndex(r + 7)];
	}

	protected void decompressToRangeNoOff(double[] c, int rl, int ru, double[] values) {
		final int h = (ru - rl) % 8;
		for(int rc = rl; rc < rl + h; rc++)
			c[rc] += values[getIndex(rc)];
		for(int rc = rl + h; rc < ru; rc += 8)
			decompressToRangeNoOffBy8(c, rc, values);
	}

	/**
	 * Split this mapping into x smaller mappings according to round robin.
	 * 
	 * @param multiplier The number of smaller mappings to construct
	 * @return The list of smaller mappings
	 */
	public AMapToData[] splitReshapeDDC(final int multiplier) {

		final int s = size();
		final AMapToData[] ret = new AMapToData[multiplier];
		final int eachSize = s / multiplier;
		for(int i = 0; i < multiplier; i++)
			ret[i] = MapToFactory.create(eachSize, getUnique());

		final int blkz = Math.max(eachSize / 8, 2048) * multiplier;
		for(int i = 0; i < s; i += blkz)
			splitReshapeDDCBlock(ret, multiplier, i, Math.min(i + blkz, s));

		return ret;
	}

	public AMapToData[] splitReshapeDDCPushDown(final int multiplier, final ExecutorService pool) throws Exception {

		final int s = size();
		final AMapToData[] ret = new AMapToData[multiplier];
		final int eachSize = s / multiplier;
		for(int i = 0; i < multiplier; i++)
			ret[i] = MapToFactory.create(eachSize, getUnique());

		final int blkz = Math.max(eachSize / 8, 2048) * multiplier;
		List<Future<?>> tasks = new ArrayList<>();
		for(int i = 0; i < s; i += blkz) {
			final int start = i;
			final int end = Math.min(i + blkz, s);
			tasks.add(pool.submit(() -> splitReshapeDDCBlock(ret, multiplier, start, end)));
		}

		for(Future<?> t : tasks)
			t.get();

		return ret;
	}

	private void splitReshapeDDCBlock(final AMapToData[] ret, final int multiplier, final int start, final int end) {

		for(int i = start; i < end; i += multiplier)
			splitReshapeDDCRow(ret, multiplier, i);
	}

	private void splitReshapeDDCRow(final AMapToData[] ret, final int multiplier, final int i) {
		final int off = i / multiplier;
		final int end = i + multiplier;
		for(int j = i; j < end; j++)
			ret[j % multiplier].set(off, getIndex(j));
	}

	@Override
	public String toString() {
		final int sz = size();
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		for(int i = 0; i < sz - 1; i++)
			sb.append(getIndex(i) + ", ");
		sb.append(getIndex(sz - 1));
		sb.append("]");
		return sb.toString();
	}
}
