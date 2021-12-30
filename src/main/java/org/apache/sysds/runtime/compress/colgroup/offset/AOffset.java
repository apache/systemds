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
package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Offset list encoder interface.
 * 
 * It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
 * 
 * The no duplicate is important since 0 values are exploited to encode an offset of max representable value + 1. This
 * gives the ability to encode data, where the offsets are greater than the available highest value that can be
 * represented size.
 */
public abstract class AOffset implements Serializable {
	private static final long serialVersionUID = 6910025321078561338L;

	protected static final Log LOG = LogFactory.getLog(AOffset.class.getName());

	/** Thread local cache for a single recently used Iterator, this is used for cache blocking */
	private ThreadLocal<OffsetCache> cacheRow = new ThreadLocal<OffsetCache>() {
		@Override
		protected OffsetCache initialValue() {
			return null;
		}
	};

	/** Memorizer for the row indexes mostly used for when we parallelize across rows */
	private Map<Integer, AIterator> memorizer = null;

	/**
	 * Get an iterator of the offsets.
	 * 
	 * @return AIterator that iterate through index and dictionary offset values.
	 */
	public abstract AIterator getIterator();

	/**
	 * Get an iterator that is pointing at a specific offset.
	 * 
	 * @param row The row requested.
	 * @return AIterator that iterate through index and dictionary offset values.
	 */
	public AIterator getIterator(int row) {
		if(row <= getOffsetToFirst())
			return getIterator();
		else if(row > getOffsetToLast())
			return null;

		// Try the cache first.
		OffsetCache c = cacheRow.get();

		if(c != null && c.row == row)
			return c.it.clone();
		else {
			if(memorizer != null && memorizer.containsKey(row))
				return memorizer.get(row).clone();
			// Use the cached iterator if it is closer to the queried row.
			AIterator it = c != null && c.row < row ? c.it.clone() : getIterator();
			it.skipTo(row);
			// cache this new iterator.
			cacheIterator(it.clone(), row);
			memorizeIterator(it.clone(), row);
			return it;
		}

	}

	/**
	 * Cache a iterator in use, note that there is no check for if the iterator is correctly positioned at the given row
	 * 
	 * @param it  The Iterator to cache
	 * @param row The row index to cache the iterator as.
	 */
	public void cacheIterator(AIterator it, int row) {
		if(it == null)
			return;
		cacheRow.set(new OffsetCache(it, row));
	}

	private void memorizeIterator(AIterator it, int row) {
		if(it == null)
			return;
		else if(memorizer == null)
			memorizer = new HashMap<>();
		memorizer.put(row, it);
	}

	/**
	 * Write the offsets to disk.
	 * 
	 * If you implement another remember to write the ordinal of the new type to disk as well and add it to the
	 * OffsetFactory.
	 * 
	 * @param out The output to write to
	 * @throws IOException Exception that happens if the IO fails to write.
	 */
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Get the offset to the first index
	 * 
	 * @return The first index offset
	 */
	public abstract int getOffsetToFirst();

	/**
	 * Get the offset to the last value
	 * 
	 * @return The last values offset
	 */
	public abstract int getOffsetToLast();

	/**
	 * Get the in memory size of the Offset object
	 * 
	 * @return In memory size as a long.
	 */
	public abstract long getInMemorySize();

	/**
	 * Remember to include the ordinal of the type of offset list.
	 * 
	 * @return the size on disk as a long.
	 */
	public abstract long getExactSizeOnDisk();

	/**
	 * Get the number of contained elements, This method iterate the entire offset list, so it is not constant lookup.
	 * 
	 * @return The number of indexes.
	 */
	public abstract int getSize();

	/**
	 * Get the length of the underlying offsets lists.
	 * 
	 * @return The number of offsets.
	 */
	public abstract int getOffsetsLength();

	public final void preAggregateDenseMap(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		char[] data) {
		// multi row iterator.
		final AIterator it = getIterator(cl);
		if(it == null)
			return;
		else if(it.offset > cu)
			cacheIterator(it, cu); // cache this iterator.
		else if(rl == ru - 1) {
			final DenseBlock db = m.getDenseBlock();
			final double[] mV = db.values(rl);
			final int off = db.pos(rl);
			preAggregateDenseMapRowChar(mV, off, preAV, cu, nVal, data, it);
		}
		else {
			final DenseBlock db = m.getDenseBlock();
			preAggregateDenseMapRowsChar(db, preAV, rl, ru, cl, cu, nVal, data, it);
		}
	}

	public final void preAggregateDenseMap(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		byte[] data) {
		// multi row iterator.
		final AIterator it = getIterator(cl);
		if(it == null)
			return;
		else if(it.offset > cu)
			cacheIterator(it, cu); // cache this iterator.
		else if(rl == ru - 1) {
			final DenseBlock db = m.getDenseBlock();
			final double[] mV = db.values(rl);
			final int off = db.pos(rl);
			preAggregateDenseMapRowByte(mV, off, preAV, cu, nVal, data, it);
		}
		else {
			final DenseBlock db = m.getDenseBlock();
			preAggregateDenseMapRowsByte(db, preAV, rl, ru, cl, cu, nVal, data, it);
		}
	}

	public final void preAggregateDenseMap(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		BitSet data) {
		// multi row iterator.
		final AIterator it = getIterator(cl);
		if(it == null)
			return;
		else if(it.offset > cu)
			cacheIterator(it, cu); // cache this iterator.
		else if(rl == ru - 1) {
			final DenseBlock db = m.getDenseBlock();
			final double[] mV = db.values(rl);
			final int off = db.pos(rl);
			preAggregateDenseMapRowBit(mV, off, preAV, cu, nVal, data, it);
		}
		else {
			final DenseBlock db = m.getDenseBlock();
			preAggregateDenseMapRowsBit(db, preAV, rl, ru, cl, cu, nVal, data, it);
		}
	}

	protected abstract void preAggregateDenseMapRowByte(double[] mV, int off, double[] preAV, int cu, int nVal,
		byte[] data, AIterator it);

	protected abstract void preAggregateDenseMapRowChar(double[] mV, int off, double[] preAV, int cu, int nVal,
		char[] data, AIterator it);

	protected abstract void preAggregateDenseMapRowBit(double[] mV, int off, double[] preAV, int cu, int nVal,
		BitSet data, AIterator it);

	protected abstract void preAggregateDenseMapRowsChar(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, char[] data, AIterator it);

	protected abstract void preAggregateDenseMapRowsByte(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, byte[] data, AIterator it);

	protected void preAggregateDenseMapRowsBit(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		BitSet data, AIterator it) {
		if(cu < getOffsetToLast() + 1)
			preAggregateDenseMapRowsBitBelowEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
		else
			preAggregateDenseMapRowsBitEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
	}

	protected void preAggregateDenseMapRowsBitBelowEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, BitSet data, AIterator it) {
		final double[] vals = db.values(rl);
		final int nCol = db.getCumODims(0);
		while(it.offset < cu) {
			final int dataOffset = data.get(it.getDataIndex()) ? 1 : 0;
			final int start = it.offset + nCol * rl;
			final int end = it.offset + nCol * ru;
			for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
				preAV[offOut] += vals[off];
			it.next();
		}

		cacheIterator(it, cu);
	}

	protected void preAggregateDenseMapRowsBitEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, BitSet data, AIterator it) {
		final double[] vals = db.values(rl);
		final int nCol = db.getCumODims(0);
		final int last = getOffsetToLast();
		int dataOffset = data.get(it.getDataIndex()) ? 1 : 0;
		int start = it.offset + nCol * rl;
		int end = it.offset + nCol * ru;
		for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
			preAV[offOut] += vals[off];
		while(it.offset < last) {
			it.next();
			dataOffset = data.get(it.getDataIndex()) ? 1 : 0;
			start = it.offset + nCol * rl;
			end = it.offset + nCol * ru;
			for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
				preAV[offOut] += vals[off];
		}
	}

	public final void preAggregateSparseMap(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, char[] data) {
		final AIterator it = getIterator();
		if(rl == ru - 1)
			preAggregateSparseMapRow(sb, preAV, rl, nVal, data, it);
		else
			throw new NotImplementedException("MultiRow Preaggregation not supported yet");
	}

	public final void preAggregateSparseMap(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, byte[] data) {
		final AIterator it = getIterator();
		if(rl == ru - 1)
			preAggregateSparseMapRow(sb, preAV, rl, nVal, data, it);
		else
			throw new NotImplementedException("MultiRow Preaggregation not supported yet");
	}

	public final void preAggregateSparseMap(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, BitSet data) {
		final AIterator it = getIterator();
		if(rl == ru - 1)
			preAggregateSparseMapRow(sb, preAV, rl, nVal, data, it);
		else
			throw new NotImplementedException("MultiRow Preaggregation not supported yet");
	}

	private void preAggregateSparseMapRow(SparseBlock sb, double[] preAV, int r, int nVal, byte[] data, AIterator it) {
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);

		final int maxId = data.length - 1;

		int j = apos;
		while(j < alen) {
			if(aix[j] == it.offset) {
				preAV[data[it.getDataIndex()] & 0xFF] += avals[j++];
				if(it.getDataIndex() >= maxId)
					break;
				it.next();
			}
			else if(aix[j] < it.offset) {
				j++;
			}
			else {
				if(it.getDataIndex() >= maxId)
					break;
				it.next();
			}
		}
	}

	private void preAggregateSparseMapRow(SparseBlock sb, double[] preAV, int r, int nVal, char[] data, AIterator it) {
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);

		final int maxId = data.length - 1;
		int j = apos;
		while(j < alen) {
			if(aix[j] == it.offset) {
				preAV[data[it.getDataIndex()]] += avals[j++];
				if(it.getDataIndex() >= maxId)
					break;
				it.next();
			}
			else if(aix[j] < it.offset) {
				j++;
			}
			else {
				if(it.getDataIndex() >= maxId)
					break;
				it.next();
			}
		}
	}

	private void preAggregateSparseMapRow(SparseBlock sb, double[] preAV, int r, int nVal, BitSet data, AIterator it) {
		final int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);
		final int last = getOffsetToLast();

		int j = apos;
		while(it.offset < last && j < alen) {
			if(aix[j] == it.offset) {
				preAV[data.get(it.getDataIndex()) ? 1 : 0] += avals[j++];
				it.next();
			}
			if(j < alen)
				while(it.offset < last && aix[j] > it.offset)
					it.next();
			while(j < alen && aix[j] < it.offset)
				j++;
		}
		while(j < alen && aix[j] < it.offset)
			j++;
		if(j != alen && aix[j] == it.offset)
			preAV[data.get(it.getDataIndex()) ? 1 : 0] += avals[j];

	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		final AIterator it = getIterator();
		final int last = getOffsetToLast();
		sb.append("[");
		while(it.offset < last) {
			sb.append(it.offset);
			sb.append(", ");
			it.next();
		}
		sb.append(it.offset);
		sb.append("]");
		return sb.toString();
	}

	protected static class OffsetCache {
		protected final AIterator it;
		protected final int row;

		protected OffsetCache(AIterator it, int row) {
			this.it = it;
			this.row = row;
		}
	}
}
