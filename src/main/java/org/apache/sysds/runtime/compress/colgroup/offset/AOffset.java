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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
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

	/**
	 * Get an iterator of the offsets while also maintaining the data index pointer.
	 * 
	 * @return AIterator that iterate through index and dictionary offset values.
	 */
	public abstract AIterator getIterator();

	/**
	 * Get an OffsetIterator of current offsets not maintaining the data index.
	 * 
	 * @return AIterator that iterator through the delta offsets.
	 */
	public abstract AOffsetIterator getOffsetIterator();

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
			AIterator it = null;
			// Use the cached iterator if it is closer to the queried row.
			it = c != null && c.row < row ? c.it.clone() : getIterator();
			it.skipTo(row);
			// cache this new iterator.
			cacheIterator(it.clone(), row);
			return it;
		}
	}

	/**
	 * Get an iterator that is pointing only at offsets from specific offset
	 * 
	 * @param row The row requested.
	 * @return AOffsetIterator that iterate through index only offset values.
	 */
	public AOffsetIterator getOffsetIterator(int row) {
		AOffsetIterator it = getOffsetIterator();
		final int last = Math.min(row, getOffsetToLast());
		while(it.value() < last)
			it.next();
		return it;
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
		AMapToData data) {
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
			// guaranteed contiguous.
			preAggregateDenseMapRow(mV, off, preAV, cu, nVal, data, it);
		}
		else {
			final DenseBlock db = m.getDenseBlock();
			preAggregateDenseMapRows(db, preAV, rl, ru, cl, cu, nVal, data, it);
		}
	}

	protected final void preAggregateDenseMapRow(double[] mV, int off, double[] preAV, int cu, int nVal, AMapToData data,
		AIterator it) {
		final int last = getOffsetToLast();
		if(cu <= last)
			preAggregateDenseMapRowBellowEnd(mV, off, preAV, cu, nVal, data, it);
		else
			preAggregateDenseMapRowEnd(mV, off, preAV, last, nVal, data, it);
	}

	protected final void preAggregateDenseMapRowBellowEnd(final double[] mV, final int off, final double[] preAV, int cu,
		final int nVal, final AMapToData data, final AIterator it) {
		it.offset += off;
		cu += off;
		while(it.offset < cu) {
			preAV[data.getIndex(it.getDataIndex())] += mV[it.offset];
			it.next();
		}
		it.offset -= off;
		cu -= off;
		cacheIterator(it, cu);
	}

	protected final void preAggregateDenseMapRowEnd(final double[] mV, final int off, final double[] preAV,
		final int last, final int nVal, final AMapToData data, final AIterator it) {

		while(it.offset < last) {
			final int dx = it.getDataIndex();
			preAV[data.getIndex(dx)] += mV[off + it.offset];
			it.next();
		}
		preAV[data.getIndex(it.getDataIndex())] += mV[off + last];
	}

	protected final void preAggregateDenseMapRows(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, AMapToData data, AIterator it) {
		if(!db.isContiguous())
			throw new NotImplementedException("Not implemented support for preAggregate non contiguous dense matrix");
		else if(cu <= getOffsetToLast())
			preAggregateDenseMapRowsBelowEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
		else
			preAggregateDenseMapRowsEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
	}

	private void preAggregateDenseMapRowsBelowEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, AMapToData data, AIterator it) {
		final double[] vals = db.values(rl);
		final int nCol = db.getCumODims(0);
		while(it.offset < cu) {
			final int dataOffset = data.getIndex(it.getDataIndex());
			final int start = it.offset + nCol * rl;
			final int end = it.offset + nCol * ru;
			for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
				preAV[offOut] += vals[off];
			it.next();
		}
		cacheIterator(it, cu);
	}

	private void preAggregateDenseMapRowsEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		AMapToData data, AIterator it) {
		final double[] vals = db.values(rl);
		final int nCol = db.getCumODims(0);
		final int last = getOffsetToLast();
		int dataOffset = data.getIndex(it.getDataIndex());
		int start = it.offset + nCol * rl;
		int end = it.offset + nCol * ru;
		for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
			preAV[offOut] += vals[off];
		while(it.offset < last) {
			it.next();
			dataOffset = data.getIndex(it.getDataIndex());
			start = it.offset + nCol * rl;
			end = it.offset + nCol * ru;
			for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
				preAV[offOut] += vals[off];
		}
	}

	public final void preAggregateSparseMap(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, AMapToData data) {
		final AIterator it = getIterator();
		if(rl == ru - 1)
			preAggregateSparseMapRow(sb, preAV, rl, nVal, data, it);
		else
			preAggregateSparseMapRows(sb, preAV, rl, ru, nVal, data, it);
	}

	private void preAggregateSparseMapRow(SparseBlock sb, double[] preAV, int r, int nVal, AMapToData data,
		AIterator it) {
		if(sb.isEmpty(r))
			return;
		final int alen = sb.size(r) + sb.pos(r);
		final int[] aix = sb.indexes(r);
		final int last = getOffsetToLast();
		if(aix[alen - 1] < last)
			preAggregateSparseMapRowBellowEnd(sb, preAV, r, nVal, data, it);
		else
			preAggregateSparseMapRowEnd(sb, preAV, r, nVal, data, it);
	}

	private final void preAggregateSparseMapRowBellowEnd(SparseBlock sb, double[] preAV, int r, int nVal,
		AMapToData data, AIterator it) {
		int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);
		int v = it.value();
		while(apos < alen) {
			if(aix[apos] == v) {
				preAV[data.getIndex(it.getDataIndex())] += avals[apos++];
				v = it.next();
			}
			else if(aix[apos] < v)
				apos++;
			else
				v = it.next();
		}
	}

	private final void preAggregateSparseMapRowEnd(SparseBlock sb, double[] preAV, int r, int nVal, AMapToData data,
		AIterator it) {
		int apos = sb.pos(r);
		final int alen = sb.size(r) + apos;
		final int[] aix = sb.indexes(r);
		final double[] avals = sb.values(r);
		final int last = getOffsetToLast();
		int v = it.value();
		while(v < last) {
			if(aix[apos] == v) {
				preAV[data.getIndex(it.getDataIndex())] += avals[apos++];
				v = it.next();
			}
			else if(aix[apos] < v)
				apos++;
			else
				v = it.next();
		}
		while(aix[apos] < last && apos < alen)
			apos++;
		if(v == aix[apos]) // process last element
			preAV[data.getIndex(it.getDataIndex())] += avals[apos];
	}

	private void preAggregateSparseMapRows(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, AMapToData data,
		AIterator it) {
		int i = it.value();
		final int last = getOffsetToLast();
		final int[] aOffs = new int[ru - rl];
		for(int r = rl; r < ru; r++)
			aOffs[r - rl] = sb.pos(r);

		while(i < last) { // while we are not done iterating
			for(int r = rl; r < ru; r++) {
				if(sb.isEmpty(r))
					continue;
				final int off = r - rl;
				int apos = aOffs[off]; // current offset
				final int alen = sb.size(r) + sb.pos(r);
				final int[] aix = sb.indexes(r);
				while(apos < alen && aix[apos] < i)// increment all pointers to offset
					apos++;

				if(apos < alen && aix[apos] == i)
					preAV[off * nVal + data.getIndex(it.getDataIndex())] += sb.values(r)[apos];
				aOffs[off] = apos;
			}
			i = it.next();
		}

		// process final element
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int off = r - rl;
			int apos = aOffs[off];
			final int alen = sb.size(r) + sb.pos(r);
			final int[] aix = sb.indexes(r);
			while(apos < alen && aix[apos] < last)
				apos++;

			if(apos < alen && aix[apos] == last)
				preAV[off * nVal + data.getIndex(it.getDataIndex())] += sb.values(r)[apos];
			aOffs[off] = apos;
		}
	}

	public boolean equals(AOffset b) {
		if(getOffsetToLast() == b.getOffsetToLast()) {
			int last = getOffsetToLast();
			AOffsetIterator ia = getOffsetIterator();
			AOffsetIterator ib = b.getOffsetIterator();
			while(ia.value() < last) {
				if(ia.value() != ib.value())
					return false;
				ia.next();
				ib.next();
				if(ib.value() == last && ia.value() != last)
					return false;
			}
			return true;
		}
		return false;
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

		if(it.offset != last)
			throw new DMLCompressionException(
				"Invalid iteration of offset when making string, the last offset is not equal to a iteration: "
					+ getOffsetToLast() + " String: " + sb.toString());
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
