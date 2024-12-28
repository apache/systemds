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
import java.lang.ref.SoftReference;
import java.util.Arrays;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AOffsetsGroup;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToUByte;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;

/**
 * Offset list encoder abstract class.
 * <p>
 * It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
 * </p>
 * <p>
 * The no duplicate is important since 0 values are exploited to encode an offset of max representable value + 1. This
 * gives the ability to encode data, where the offsets are greater than the available highest value that can be
 * represented size.
 * </p>
 */
public abstract class AOffset implements Serializable {
	private static final long serialVersionUID = 6910025321078561338L;

	protected static final Log LOG = LogFactory.getLog(AOffset.class.getName());

	/** Cached final empty slice to return in cases of empty slice returns to avoid object allocation */
	protected static final OffsetSliceInfo EMPTY_SLICE = new OffsetSliceInfo(-1, -1, new OffsetEmpty());

	/** The skip list stride size, aka how many indexes skipped for each index. */
	protected static final int SKIP_STRIDE = 1000;

	/** SoftReference of the skip list to be dematerialized on memory pressure */
	private volatile SoftReference<OffsetCacheV2[]> skipList = null;

	/** Thread local cache for a single recently used Iterator, this is used for cache blocking */
	private volatile ThreadLocal<OffsetCache> cacheRow = null;

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

	private AIterator getIteratorFromSkipList(OffsetCacheV2 c) {
		return getIteratorFromIndexOff(c.row, c.dataIndex, c.offIndex);
	}

	protected abstract AIterator getIteratorFromIndexOff(int row, int dataIndex, int offIdx);

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

		final OffsetCache c;
		if(getLength() < SKIP_STRIDE)
			c = null;
		else if(cacheRow == null)
			c = null;
		else
			c = cacheRow.get();

		if(c != null && c.row == row)
			return c.it.clone();
		else if(getLength() < SKIP_STRIDE)
			return getIteratorSmallOffset(row);
		else
			return getIteratorLargeOffset(row);
	}

	/**
	 * Get an iterator that is pointing to a specific offset, this method skips looking at our cache of iterators.
	 * 
	 * @param row The row to look at
	 * @return The iterator associated with the row.
	 */
	private AIterator getIteratorSkipCache(int row) {
		if(row <= getOffsetToFirst())
			return getIterator();
		else if(row > getOffsetToLast())
			return null;
		else if(getLength() < SKIP_STRIDE)
			return getIteratorSmallOffset(row);
		else
			return getIteratorLargeOffset(row);
	}

	private AIterator getIteratorSmallOffset(int row) {
		AIterator it = getIterator();
		it.skipTo(row);
		cacheIterator(it.clone(), row);
		return it;
	}

	private final AIterator getIteratorLargeOffset(int row) {
		if(skipList == null || skipList.get() == null)
			constructSkipList();
		final OffsetCacheV2[] skip = skipList.get();

		// guaranteed not to go over limit of skip list.
		int idx = 0;
		while(idx < skip.length //
			&& skip[idx] != null //
			&& skip[idx].row <= row)
			idx++;

		final AIterator it = idx == 0 ? getIterator() : getIteratorFromSkipList(skip[idx - 1]);
		it.skipTo(row);
		cacheIterator(it.clone(), row);
		return it;
	}

	public synchronized void constructSkipList() {
		if(skipList != null && skipList.get() != null)
			return;

		// not actual accurate but applicable.
		final int last = getOffsetToLast();
		final int skipSize = last / SKIP_STRIDE + 1;
		if(skipSize == 1)
			return; // do not construct the skip if the size is less than SKIP_STRIDE

		final OffsetCacheV2[] skipListTmp = new OffsetCacheV2[skipSize];
		final AIterator it = getIterator();

		int skipListIdx = 0;
		while(it.value() < last) { // guaranteed not to go over skipListTmpLength
			int next = skipListIdx * SKIP_STRIDE + SKIP_STRIDE;
			while(it.value() < next && it.value() < last)
				it.next();
			skipListTmp[skipListIdx++] = new OffsetCacheV2(it.value(), it.getDataIndex(), it.getOffsetsIndex());
		}

		skipList = new SoftReference<>(skipListTmp);
	}

	public synchronized void clearSkipList() {
		if(skipList != null)
			skipList.clear();
	}

	/**
	 * Cache a iterator in use, note that there is no check for if the iterator is correctly positioned at the given row
	 * 
	 * @param it  The Iterator to cache
	 * @param row The row index to cache the iterator as.
	 */
	public void cacheIterator(AIterator it, int row) {
		if(it == null || getLength() < SKIP_STRIDE)
			return;

		if(cacheRow == null) {
			cacheRow = new ThreadLocal<>() {
				@Override
				protected OffsetCache initialValue() {
					return new OffsetCache(it, row);
				}
			};
		}
		else {
			cacheRow.set(new OffsetCache(it, row));
		}
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
	 * Pre aggregate the specified row and block range from a dense MatrixBlock to prepare for compressed multiplication.
	 * 
	 * @param db    The DenseBlock to extract values from.
	 * @param preAV The pre aggregate row linearized double array to put the values into.
	 * @param rl    The row lower to start from (this is referring to the left matrix of the multiplication)
	 * @param ru    The row upper to end at (not inclusive) (this is referring to the left matrix of the multiplication)
	 * @param cl    The column lower to start at (this is referring to the right matrix of the multiplication)
	 * @param cu    The column upper to end at (not inclusive) (this is referring to the right matrix of the
	 *              multiplication)
	 * @param nVal  The number of distinct values in the PreAV indicating number of columns in the Pre aggregate
	 * @param data  The mapping to column positions in the preAV
	 */
	public final void preAggregateDenseMap(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		AMapToData data) {
		// multi row iterator.
		final AIterator it = getIterator(cl);
		if(it == null)
			return;
		else if(it.offset > cu)
			cacheIterator(it, cu); // cache this iterator.
		else if(rl == ru - 1) {
			final double[] mV = db.values(rl);
			final int off = db.pos(rl);
			// guaranteed contiguous.
			preAggDenseMapSingleRow(mV, off, preAV, cu, nVal, data, it);
		}
		else {
			preAggDenseMapMultiRows(db, preAV, rl, ru, cl, cu, nVal, data, it);
		}
	}

	private final void preAggDenseMapSingleRow(double[] mV, int off, double[] preAV, int cu, int nVal, AMapToData data,
		AIterator it) {
		final int last = getOffsetToLast();
		if(cu <= last)
			preAggDenseMapRowBellowEnd(mV, off, preAV, cu, data, it);
		else
			preAggDenseMapSingleRowEnd(mV, off, preAV, last, data, it);
	}

	private final void preAggDenseMapRowBellowEnd(final double[] mV, final int off, final double[] preAV, int cu,
		final AMapToData data, final AIterator it) {
		// Increment and prepare iterator.
		it.offset += off;
		cu += off;
		preAggDenseMapRowBE(mV, preAV, cu, data, it);
		// Decrement iterator for next call.
		it.offset -= off;
		cu -= off;
		cacheIterator(it, cu);
	}

	private final void preAggDenseMapRowBE(final double[] mV, final double[] preAV, final int cu, final AMapToData data,
		final AIterator it) {
		if(data instanceof MapToUByte)
			preAggDenseMapRowBE_UByte(mV, preAV, cu, (MapToUByte) data, it);
		else if(data instanceof MapToByte)
			preAggDenseMapRowBE_Byte(mV, preAV, cu, (MapToByte) data, it);
		else if(data instanceof MapToChar)
			preAggDenseMapRowBE_Char(mV, preAV, cu, (MapToChar) data, it);
		else
			preAggDenseMapRowBE_Generic(mV, preAV, cu, data, it);
	}

	private final void preAggDenseMapRowBE_UByte(final double[] mV, final double[] preAV, final int cu,
		final MapToUByte data, final AIterator it) {
		// for JIT Compilation
		while(it.offset < cu) {
			preAV[data.getIndex(it.getDataIndex())] += mV[it.offset];
			it.next();
		}
	}

	private final void preAggDenseMapRowBE_Byte(final double[] mV, final double[] preAV, final int cu,
		final MapToByte data, final AIterator it) {
		// for JIT Compilation
		while(it.offset < cu) {
			preAV[data.getIndex(it.getDataIndex())] += mV[it.offset];
			it.next();
		}
	}

	private final void preAggDenseMapRowBE_Char(final double[] mV, final double[] preAV, final int cu,
		final MapToChar data, final AIterator it) {
		// for JIT Compilation
		while(it.offset < cu) {
			preAV[data.getIndex(it.getDataIndex())] += mV[it.offset];
			it.next();
		}
	}

	private final void preAggDenseMapRowBE_Generic(final double[] mV, final double[] preAV, final int cu,
		final AMapToData data, final AIterator it) {
		// for JIT Compilation
		while(it.offset < cu) {
			preAV[data.getIndex(it.getDataIndex())] += mV[it.offset];
			it.next();
		}
	}

	private final void preAggDenseMapSingleRowEnd(final double[] mV, final int off, final double[] preAV, final int last,
		final AMapToData data, final AIterator it) {

		while(it.offset < last) {
			final int dx = it.getDataIndex();
			preAV[data.getIndex(dx)] += mV[off + it.offset];
			it.next();
		}
		preAV[data.getIndex(it.getDataIndex())] += mV[off + last];
	}

	private final void preAggDenseMapMultiRows(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		AMapToData data, AIterator it) {
		if(!db.isContiguous())
			throw new NotImplementedException("Not implemented support for preAggregate non contiguous dense matrix");
		else if(cu <= getOffsetToLast())
			preAggDenseMapMultiRowsBelowEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
		else
			preAggDenseMapMultiRowsEnd(db, preAV, rl, ru, cl, cu, nVal, data, it);
	}

	private final void preAggDenseMapMultiRowsBelowEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
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

	private final void preAggDenseMapMultiRowsEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, AMapToData data, AIterator it) {
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

	public final void preAggSparseMap(SparseBlock sb, double[] preAV, int rl, int ru, int nVal, AMapToData data) {
		final AIterator it = getIterator();
		if(rl == ru - 1)
			preAggSparseMapSingleRow(sb, preAV, rl, nVal, data, it);
		else
			preAggSparseMapMultipleRows(sb, preAV, rl, ru, nVal, data, it);
	}

	private final void preAggSparseMapSingleRow(SparseBlock sb, double[] preAV, int r, int nVal, AMapToData data,
		AIterator it) {
		if(sb.isEmpty(r))
			return;
		final int alen = sb.size(r) + sb.pos(r);
		final int[] aix = sb.indexes(r);
		final int last = getOffsetToLast();
		if(aix[alen - 1] < last)
			preAggSparseMapRowBellowEnd(sb, preAV, r, nVal, data, it);
		else
			preAggSparseMapRowEnd(sb, preAV, r, nVal, data, it);
	}

	private final void preAggSparseMapRowBellowEnd(SparseBlock sb, double[] preAV, int r, int nVal, AMapToData data,
		AIterator it) {
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

	private final void preAggSparseMapRowEnd(SparseBlock sb, double[] preAV, int r, int nVal, AMapToData data,
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

	private final void preAggSparseMapMultipleRows(final SparseBlock sb, final double[] preAV, final int rl,
		final int ru, final int nVal, final AMapToData data, AIterator it) {
		int i = it.value();
		final int last = getOffsetToLast();
		final int[] aOffs = new int[ru - rl];
		for(int r = rl; r < ru; r++)
			aOffs[r - rl] = sb.pos(r);

		while(i < last) { // while we are not done iterating
			preAggSparseMapRow(sb, preAV, rl, ru, nVal, data.getIndex(it.getDataIndex()), i, aOffs);
			i = it.next();
		}

		preAggSparseMapRow(sb, preAV, rl, ru, nVal, data.getIndex(it.getDataIndex()), last, aOffs);
	}

	private final void preAggSparseMapRow(final SparseBlock sb, final double[] preAV, final int rl, final int ru,
		final int nVal, final int dataIndex, final int i, final int[] aOffs) {

		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int off = r - rl;
			int apos = aOffs[off]; // current offset
			final int alen = sb.size(r) + sb.pos(r);
			final int[] aix = sb.indexes(r);
			final double[] avals = sb.values(r);
			while(apos < alen && aix[apos] < i)// increment all pointers to offset
				apos++;

			if(apos < alen && aix[apos] == i)
				preAV[off * nVal + dataIndex] += avals[apos];
			aOffs[off] = apos;
		}
	}

	@Override
	public boolean equals(Object o) {
		return o instanceof AOffset && this.equals((AOffset) o);
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

	/**
	 * Move the index start x cells
	 * 
	 * @param m The amount to move
	 * @return The moved index.
	 */
	public abstract AOffset moveIndex(int m);

	/**
	 * Get the length of the underlying array. This does not reflect the number of contained elements, since some of the
	 * elements can be skips.
	 * 
	 * @return The length of the underlying arrays
	 */
	public abstract int getLength();

	/**
	 * Slice the offsets based on the specified range
	 * 
	 * @param l inclusive lower bound
	 * @param u exclusive upper bound
	 * @return The slice info containing the new slice.
	 */
	public OffsetSliceInfo slice(int l, int u) {
		final int first = getOffsetToFirst();
		final int last = getOffsetToLast();
		final int s = getSize();

		if(l <= first && u > last) {
			if(l == 0)
				return new OffsetSliceInfo(0, s, this);
			else
				return new OffsetSliceInfo(0, s, moveIndex(l));
		}
		else if (u < first)
			return EMPTY_SLICE;

		final AIterator it = getIteratorSkipCache(l);

		if(it == null || it.value() >= u)
			return EMPTY_SLICE;

		if(u >= last) // If including the last do not iterate.
			return constructSliceReturn(l, u, it.getDataIndex(), s - 1, it.getOffsetsIndex(), getLength(), it.value(),
				last);
		else // Have to iterate through until we find last.
			return genericSlice(l, u, it);
	}

	private OffsetSliceInfo genericSlice(int l, int u, AIterator it) {
		// point at current one should be guaranteed inside the range.
		final int low = it.getDataIndex();
		final int lowOff = it.getOffsetsIndex();
		final int lowValue = it.value();

		// set c
		int high = low;
		int highOff = lowOff;
		int highValue = lowValue;
		while(it.value() < u) {
			// TODO add previous command that would allow us to simplify this loop.

			highValue = it.value();
			high = it.getDataIndex();
			highOff = it.getOffsetsIndex();

			it.next();
		}

		return constructSliceReturn(l, u, low, high, lowOff, highOff, lowValue, highValue);

	}

	private final OffsetSliceInfo constructSliceReturn(int l, int u, int low, int high, int lowOff, int highOff,
		int lowValue, int highValue) {
		if(low == high) // Implicit lowValue == highValue
			return new OffsetSliceInfo(low, high + 1, new OffsetSingle(lowValue - l));
		else if(low + 1 == high)
			return new OffsetSliceInfo(low, high + 1, new OffsetTwo(lowValue - l, highValue - l));
		else
			return ((ISliceOffset) this).slice(lowOff, highOff, lowValue - l, highValue - l, low, high);
	}

	/**
	 * Append the offsets from that other offset to the offsets in this.
	 * 
	 * @param t That offset/
	 * @param s The total length encoded in this offset.
	 * @return this offsets followed by thats offsets.
	 */
	public AOffset append(AOffset t, int s) {
		final IntArrayList r = new IntArrayList(getLength() + t.getLength());
		final AOffsetIterator of = getOffsetIterator();
		while(of.value() < getOffsetToLast()) {
			r.appendValue(of.value());
			of.next();
		}
		r.appendValue(of.value());

		AOffsetIterator tof = t.getOffsetIterator();
		final int last = t.getOffsetToLast();
		while(tof.value() < last) {
			r.appendValue(tof.value() + s);
			tof.next();
		}
		r.appendValue(tof.value() + s);

		return OffsetFactory.createOffset(r);
	}

	/**
	 * Append a list of offsets together in order.
	 * 
	 * @param g The offsets to append together (note fist entry is equal to this)
	 * @param s The standard size of each g (except last, but that does not matter)
	 * @return The combined offsets.
	 */
	public AOffset appendN(AOffsetsGroup[] g, int s) {
		int l = 0;
		for(AOffsetsGroup gs : g)
			l += gs.getOffsets().getLength();

		IntArrayList r = new IntArrayList(l); // rough but good.

		int ss = 0;
		for(AOffsetsGroup gs : g) {
			final AOffset tof = gs.getOffsets();
			if(!(tof instanceof OffsetEmpty)) {
				final AOffsetIterator tofit = tof.getOffsetIterator();
				final int last = tof.getOffsetToLast() + ss;
				int v = tofit.value() + ss;
				while(v < last) {
					r.appendValue(v);
					v = tofit.next() + ss;
				}
				r.appendValue(v);
			}
			ss += s;
		}

		return OffsetFactory.createOffset(r);
	}

	/**
	 * Verify that the contained AOffset is a certain size, and not bigger when iterating though it.
	 * 
	 * @param size The max correct size.
	 */
	public void verify(int size) {
		AIterator it = getIterator();
		if(it != null) {
			final int last = getOffsetToLast();
			if(it.getDataIndex() > size)
				throw new DMLCompressionException("Invalid index");
			while(it.value() < last) {
				it.next();
				if(it.getDataIndex() > size) // the last index is to high
					throw new DMLCompressionException("Invalid index");
			}
		}
		else {
			if(size != 0)
				throw new DMLCompressionException("Invalid index");
		}
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		final AIterator it = getIterator();
		if(it != null) {
			final int last = getOffsetToLast();
			sb.append("[");
			sb.append(it.offset);
			while(it.offset < last) {
				it.next();
				sb.append(", ");
				sb.append(it.offset);
			}
			sb.append("]");
		}
		if(CompressedMatrixBlock.debug) {
			if(cacheRow != null && cacheRow.get() != null) {
				sb.append("\nOffset CacheRow: ");
				sb.append(cacheRow.get().toString());
			}
			if(skipList != null && skipList.get() != null) {
				sb.append("\nSkipList:");
				sb.append(Arrays.toString(skipList.get()));
			}
		}
		return sb.toString();
	}

	/**
	 * Reverse the locations of the offsets such that the inverse offsets are returned.
	 * 
	 * This means that for instance if the current offsets were 1, 3 and 5 in a 5 long list, we return 0, 2 and 4.
	 * 
	 * @param numRows The total number of rows to be contained, This should be greater or equal to last.
	 * @return The reverse offsets.
	 */
	public AOffset reverse(int numRows) {
		final int last = getOffsetToLast();
		if(numRows < last) {
			throw new DMLRuntimeException("Invalid number of rows for reverse: last: " + last + " numRows: " + numRows);
		}

		int[] newOff = new int[numRows - getSize()];
		final AOffsetIterator it = getOffsetIterator();
		int i = 0;
		int j = 0;

		while(i < last) {
			if(i == it.value()) {
				i++;
				it.next();
			}
			else
				newOff[j++] = i++;
		}
		i++; // last
		while(i < numRows)
			newOff[j++] = i++;

		return OffsetFactory.createOffset(newOff);
	}

	/**
	 * Offset slice info containing the start and end index an offset that contains the slice, and an new AOffset
	 * containing only the sliced elements
	 */
	public static final class OffsetSliceInfo {
		public final int lIndex;
		public final int uIndex;
		public final AOffset offsetSlice;

		protected OffsetSliceInfo(int l, int u, AOffset off) {
			this.lIndex = l;
			this.uIndex = u;
			this.offsetSlice = off;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("sliceInfo: ");
			sb.append(lIndex);
			sb.append("->");
			sb.append(uIndex);
			sb.append("  --  ");
			sb.append(offsetSlice);
			return sb.toString();
		}

	}

	private static class OffsetCache {
		private final AIterator it;
		private final int row;

		private OffsetCache(AIterator it, int row) {
			this.it = it;
			this.row = row;
		}

		@Override
		public String toString() {
			return "r " + row + " i" + it + "\n";
		}
	}

	private static class OffsetCacheV2 {
		private final int row;
		private final int offIndex;
		private final int dataIndex;

		private OffsetCacheV2(int row, int dataIndex, int offIndex) {
			this.row = row;
			this.dataIndex = dataIndex;
			this.offIndex = offIndex;
		}

		@Override
		public String toString() {
			return "r" + row + " d " + dataIndex + " o " + offIndex + "\n";
		}
	}
}
