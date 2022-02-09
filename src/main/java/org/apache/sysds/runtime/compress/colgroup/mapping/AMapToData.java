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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
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
	 * Set the index to the value.
	 * 
	 * NOTE! The value should be representable inside the map. This requirement is not checked.
	 * 
	 * @param n index to set.
	 * @param v the value to set it to.
	 */
	public abstract void set(int n, int v);

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
		off += cl;
		for(int rc = cl; rc < cu; rc++, off++)
			preAV[getIndex(rc)] += mV[off];
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
			for(int c = cl; c < cu; c++) {
				final int idx = getIndex(c);
				final int start = c + nCol * rl;
				final int end = c + nCol * ru;
				for(int offOut = idx, off = start; off < end; offOut += nVal, off += nCol) {
					preAV[offOut] += mV[off];
				}
			}
		}
		else
			throw new NotImplementedException();
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
	public abstract void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu,
		AOffset indexes);

	/**
	 * PreAggregate the SparseBlock in the range of rows given.
	 * 
	 * @param sb      The SparseBlock to preAggregate
	 * @param preAV   The target double array to put the preAggregate into
	 * @param rl      The row to start at
	 * @param ru      The row to end at (not inclusive)
	 * @param indexes The Offset Indexes to iterate through
	 */
	public abstract void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes);

	/**
	 * Get the number of counts of each unique value contained in this map. Note that in the case the mapping is shorter
	 * than number of rows the counts sum to the number of mapped values not the number of rows.
	 * 
	 * @param counts The object to return.
	 * @return the Counts
	 */
	public int[] getCounts(int[] counts) {
		final int sz = size();
		for(int i = 0; i < sz; i++)
			counts[getIndex(i)]++;
		return counts;
	}

	/**
	 * PreAggregate into dictionary with two sides of DDC.
	 * 
	 * @param tm   Map of other side
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns
	 */
	public final void preAggregateDDC_DDC(AMapToData tm, ADictionary td, Dictionary ret, int nCol) {
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
		for(int r = 0; r < sz; r++)
			v[getIndex(r)] += td[tm.getIndex(r)];
	}

	/**
	 * PreAggregate into dictionary with two sides of DDC guaranteed to multiple column tuples.
	 * 
	 * @param tm   Map of other side
	 * @param td   Dictionary to take values from (other side dictionary)
	 * @param ret  The output dictionary to aggregate into
	 * @param nCol The number of columns
	 */
	protected void preAggregateDDC_DDCMultiCol(AMapToData tm, ADictionary td, double[] v, int nCol) {
		final int sz = size();
		for(int r = 0; r < sz; r++)
			td.addToEntry(v, tm.getIndex(r), getIndex(r), nCol);
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
	public final void preAggregateDDC_SDCZ(AMapToData tm, ADictionary td, AOffset tof, Dictionary ret, int nCol) {
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

	public void preAggregateDDC_SDCZMultiCol(AMapToData tm, ADictionary td, AOffset tof, double[] v, int nCol) {
		final AOffsetIterator it = tof.getOffsetIterator();
		final int size = tm.size() - 1;
		int i = (size > 8) ? preAggregateDDC_SDCZMultiCol_vect(tm, td, tof, v, nCol, it, size) : 0;

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

	private int preAggregateDDC_SDCZMultiCol_vect(AMapToData tm, ADictionary td, AOffset tof, double[] v, int nCol,
		AOffsetIterator it, int size) {
		final int h = size % 8;
		int i = 0;
		while(i < size - h) {
			int t1 = it.value();
			int t2 = it.next();
			int t3 = it.next();
			int t4 = it.next();
			int t5 = it.next();
			int t6 = it.next();
			int t7 = it.next();
			int t8 = it.next();

			t1 = getIndex(t1);
			t2 = getIndex(t2);
			t3 = getIndex(t3);
			t4 = getIndex(t4);
			t5 = getIndex(t5);
			t6 = getIndex(t6);
			t7 = getIndex(t7);
			t8 = getIndex(t8);

			int f1 = tm.getIndex(i);
			int f2 = tm.getIndex(i + 1);
			int f3 = tm.getIndex(i + 2);
			int f4 = tm.getIndex(i + 3);
			int f5 = tm.getIndex(i + 4);
			int f6 = tm.getIndex(i + 5);
			int f7 = tm.getIndex(i + 6);
			int f8 = tm.getIndex(i + 7);

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
	public final void preAggregateSDCZ_DDC(AMapToData tm, ADictionary td, AOffset of, Dictionary ret, int nCol) {
		if(nCol == 1)
			preAggregateSDCZ_DDCSingleCol(tm, td.getValues(), of, ret.getValues());
		else
			preAggregateSDCZ_DDCMultiCol(tm, td, of, ret.getValues(), nCol);
	}

	protected void preAggregateSDCZ_DDCSingleCol(AMapToData tm, double[] td, AOffset of, double[] v) {
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int size = size() - 1;

		for(int i = 0; i < size; i++) {
			v[getIndex(i)] += td[tm.getIndex(itThis.value())];
			itThis.next();
		}
		v[getIndex(size)] += td[tm.getIndex(itThis.value())];
	}

	protected void preAggregateSDCZ_DDCMultiCol(AMapToData tm, ADictionary td, AOffset of, double[] v, int nCol) {
		final AOffsetIterator itThis = of.getOffsetIterator();
		final int size = size() - 1;

		for(int i = 0; i < size; i++) {
			td.addToEntry(v, tm.getIndex(itThis.value()), getIndex(i), nCol);
			itThis.next();
		}
		td.addToEntry(v, tm.getIndex(itThis.value()), getIndex(size), nCol);
	}

	public final void preAggregateSDCZ_SDCZ(AMapToData tm, ADictionary td, AOffset tof, AOffset of, Dictionary ret,
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
		preAggregateSDCZ_SDCZMultiCol_tail(tm, this, new Dictionary(td), dv, 1, itThat, itThis, tSize, size, i, j);
	}

	protected void preAggregateSDCZ_SDCZMultiCol(AMapToData tm, ADictionary td, AOffset tof, AOffset of, double[] dv,
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

	protected static void preAggregateSDCZ_SDCZMultiCol_tail(AMapToData tm, AMapToData m, ADictionary td, double[] dv,
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

	/**
	 * Copy the values in this map into another mapping object.
	 * 
	 * NOTE! All contained vales should be representable inside the map given. This requirement is not checked.
	 * 
	 * @param d Map to copy all values into.
	 */
	public void copy(AMapToData d) {
		final int sz = size();
		for(int i = 0; i < sz; i++)
			set(i, d.getIndex(i));
	}

	@Override
	public String toString() {
		final int sz = size();
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" [");
		for(int i = 0; i < sz - 1; i++)
			sb.append(getIndex(i) + ", ");
		sb.append(getIndex(sz - 1));
		sb.append("]");
		return sb.toString();
	}
}
