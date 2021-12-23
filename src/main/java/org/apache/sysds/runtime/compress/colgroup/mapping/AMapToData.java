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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
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
			preAggregateDenseToRow(db.values(rl), db.pos(rl), preAV, cl, cu);
		else
			preAggregateDenseRows(m, preAV, rl, ru, cl, cu);
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
	protected abstract void preAggregateDenseToRow(double[] mV, int off, double[] preAV, int cl, int cu);

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
	protected abstract void preAggregateDenseRows(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu);

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
	 * Get the number of counts of each unique value contained in this map.
	 * 
	 * @param counts The object to return.
	 * @param nRows  The number of rows in the calling column group.
	 * @return the Counts
	 */
	public int[] getCounts(int[] counts, int nRows) {
		final int nonDefaultLength = size();
		for(int i = 0; i < nonDefaultLength; i++)
			counts[getIndex(i)]++;
		counts[counts.length - 1] += nRows - nonDefaultLength;
		return counts;
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
