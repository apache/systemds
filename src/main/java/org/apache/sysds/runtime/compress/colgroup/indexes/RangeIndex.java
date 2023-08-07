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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * A Range index that contain a lower and upper bound of the indexes that is symbolize.
 * 
 * The upper bound is not inclusive
 */
public class RangeIndex extends AColIndex {
	/** Lower bound inclusive */
	private final int l;
	/** Upper bound not inclusive */
	private final int u;

	/**
	 * Construct an range index from 0 until the given nCol, not inclusive
	 * 
	 * @param nCol The upper index not included
	 */
	public RangeIndex(int nCol) {
		this(0, nCol);
	}

	/** Construct an range index */

	/**
	 * Construct an range index with lower and upper values given.
	 * 
	 * @param l lower index
	 * @param u Upper index not inclusive
	 */
	public RangeIndex(int l, int u) {
		this.l = l;
		this.u = u;

		if(l >= u)
			throw new DMLCompressionException("Invalid construction of Range Index with l: " + l + " u: " + u);
	}

	@Override
	public int size() {
		return u - l;
	}

	@Override
	public int get(int i) {
		return l + i;
	}

	@Override
	public RangeIndex shift(int i) {
		return new RangeIndex(l + i, u + i);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(ColIndexType.RANGE.ordinal());
		out.writeInt(l);
		out.writeInt(u);
	}

	public static RangeIndex read(DataInput in) throws IOException {
		int l = in.readInt();
		int u = in.readInt();
		return new RangeIndex(l, u);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4;
	}

	@Override
	public long estimateInMemorySize() {
		return estimateInMemorySizeStatic();
	}

	public static long estimateInMemorySizeStatic() {
		return 16 + 8;
	}

	@Override
	public IIterate iterator() {
		return new RangeIterator();
	}

	@Override
	public int findIndex(int i) {
		if(i < l)
			return -1;
		else if(i < u)
			return i - l;
		else
			return -1 * u - 1 + l;
	}

	@Override
	public SliceResult slice(int l, int u) {
		if(u <= this.l)
			return new SliceResult(0, 0, null);
		else if(l >= this.u)
			return new SliceResult(0, 0, null);
		else if(l <= this.l && u >= this.u)
			return new SliceResult(0, size(), new RangeIndex(this.l - l, this.u - l));
		else {
			int maxL = Math.max(l, this.l);
			int minU = Math.min(u, this.u);
			int offL = maxL - this.l;
			int offR = minU - this.l;
			return new SliceResult(offL, offR, new RangeIndex(maxL - l, minU - l ));
		}
	}

	@Override
	public boolean equals(IColIndex other) {
		if(other instanceof RangeIndex) {
			RangeIndex ot = (RangeIndex) other;
			return ot.l == l && ot.u == u;
		}
		else
			return other.equals(this);
	}

	@Override
	public IColIndex combine(IColIndex other) {
		if(other.size() == 1) {
			int v = other.get(0);
			if(v + 1 == l)
				return new RangeIndex(l - 1, u);
			else if(v == u)
				return new RangeIndex(l, u + 1);
		}
		if(other instanceof RangeIndex) {
			if(other.get(0) == u)
				return new RangeIndex(l, other.get(other.size() - 1) + 1);
			else if(other.get(other.size() - 1) == l - 1)
				return new RangeIndex(other.get(0), u);
			else if(other.get(0) < this.get(0))
				return new TwoRangesIndex((RangeIndex) other, this);
			else
				return new TwoRangesIndex(this, (RangeIndex) other);
		}

		final int sr = other.size();
		final int sl = size();
		final int[] ret = new int[sr + sl];

		int pl = 0;
		int pr = 0;
		int i = 0;
		while(pl < sl && pr < sr) {
			final int vl = get(pl);
			final int vr = other.get(pr);
			if(vl < vr) {
				ret[i++] = vl;
				pl++;
			}
			else {
				ret[i++] = vr;
				pr++;
			}
		}
		while(pl < sl)
			ret[i++] = get(pl++);
		while(pr < sr)
			ret[i++] = other.get(pr++);
		return ColIndexFactory.create(ret);
	}

	@Override
	public boolean isContiguous() {
		return true;
	}

	protected static boolean isValidRange(int[] indexes) {
		return isValidRange(indexes, indexes.length);
	}

	protected static boolean isValidRange(IntArrayList indexes) {
		return isValidRange(indexes.extractValues(), indexes.size());
	}

	private static boolean isValidRange(final int[] indexes, final int length) {
		int len = length;
		int first = indexes[0];
		int last = indexes[length - 1];

		final boolean isPossibleFistAndLast = last - first + 1 >= len;
		if(!isPossibleFistAndLast)
			throw new DMLCompressionException("Invalid Index " + Arrays.toString(indexes));
		else if(last - first + 1 == len) {
			for(int i = 1; i < length; i++)
				if(indexes[i - 1] >= indexes[i])
					throw new DMLCompressionException("Invalid Index");
			return true;
		}
		else
			return false;
	}

	@Override
	public int[] getReorderingIndex() {
		throw new DMLCompressionException("not valid to get reordering Index for range");
	}

	@Override
	public boolean isSorted() {
		return true;
	}

	@Override
	public IColIndex sort() {
		throw new DMLCompressionException("range is always sorted");
	}

	@Override
	public boolean contains(int i) {
		return l <= i && i < u;
	}

	@Override
	public double avgOfIndex() {
		double diff = u - 1 - l;
		// double s = l * diff + diff * diff * 0.5;
		// return s / diff;
		return l + diff * 0.5;
	}

	@Override
	public int hashCode() {
		return 31 * l + u;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		sb.append(l);
		sb.append(" -> ");
		sb.append(u);
		sb.append("]");
		return sb.toString();
	}

	protected class RangeIterator implements IIterate {
		int cl = l;

		@Override
		public int next() {
			return cl++;
		}

		@Override
		public boolean hasNext() {
			return cl < u;
		}

		@Override
		public int v() {
			return cl;
		}

		@Override
		public int i() {
			return cl - l;
		}
	}
}
