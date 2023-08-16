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
import java.util.stream.IntStream;

import org.apache.sysds.utils.MemoryEstimates;

public class ArrayIndex extends AColIndex {
	private final int[] cols;

	public ArrayIndex(int[] cols) {
		this.cols = cols;
	}

	@Override
	public int size() {
		return cols.length;
	}

	@Override
	public int get(int i) {
		return cols[i];
	}

	@Override
	public IColIndex shift(int i) {
		int[] ret = new int[cols.length];
		for(int j = 0; j < cols.length; j++)
			ret[j] = cols[j] + i;
		return new ArrayIndex(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(ColIndexType.ARRAY.ordinal());
		out.writeInt(cols.length);
		for(int i = 0; i < cols.length; i++)
			out.writeInt(cols[i]);
	}

	public static ArrayIndex read(DataInput in) throws IOException {
		int size = in.readInt();
		int[] cols = new int[size];
		for(int i = 0; i < size; i++)
			cols[i] = in.readInt();
		return new ArrayIndex(cols);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 * cols.length;
	}

	@Override
	public long estimateInMemorySize() {
		return estimateInMemorySizeStatic(cols.length);
	}

	public static long estimateInMemorySizeStatic(int nCol) {
		return 16 + (long) MemoryEstimates.intArrayCost(nCol);
	}

	@Override
	public IIterate iterator() {
		return new ArrayIterator();
	}

	@Override
	public int findIndex(int i) {
		return Arrays.binarySearch(cols, i);
	}

	@Override
	public boolean isContiguous() {
		return false;
	}

	@Override
	public SliceResult slice(int l, int u) {

		if(l == 0 && u > cols[cols.length - 1])
			return new SliceResult(0, cols.length, this);
		int s = Arrays.binarySearch(cols, l);
		int e = Arrays.binarySearch(cols, u);

		s = s < 0 ? Math.abs(s + 1) : s;
		e = e < 0 ? Math.abs(e + 1) : e;

		if(s == e)
			return new SliceResult(0, 0, null);

		int[] retArr = new int[e - s];
		if(l == 0)
			retArr = Arrays.copyOfRange(cols, s, e);
		else
			for(int i = s, j = 0; i < e; i++, j++)
				retArr[j] = cols[i] - l;

		SliceResult ret = new SliceResult(s, e, ColIndexFactory.create(retArr));

		return ret;
	}

	@Override
	public boolean equals(IColIndex other) {
		if(other.size() == size()) {
			if(other instanceof ArrayIndex) {
				ArrayIndex ot = (ArrayIndex) other;
				int[] otV = ot.cols;
				return Arrays.equals(cols, otV);
			}
			else if(other instanceof RangeIndex)
				return other.get(0) == cols[0] && other.get(size() - 1) == cols[size() - 1];
			else { // generic
				for(int i = 0; i < size(); i++)
					if(other.get(i) != cols[i])
						return false;
				return true;
			}
		}
		return false;
	}

	@Override
	public IColIndex combine(IColIndex other) {
		final int sr = other.size();
		final int sl = size();
		final int maxCombined = Math.max(this.get(this.size() - 1), other.get(other.size() - 1));
		final int minCombined = Math.min(this.get(0), other.get(0));
		if(sr + sl == maxCombined - minCombined + 1) {
			return new RangeIndex(minCombined, maxCombined + 1);
		}

		// LOG.error("Combining Worst " + this + " " + other);
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
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(Arrays.toString(cols));
		return sb.toString();
	}

	@Override
	public int[] getReorderingIndex() {
		int[] sortedIndices = IntStream.range(0, cols.length).boxed()//
			.sorted((i, j) -> Integer.valueOf(cols[i]).compareTo(cols[j]))//
			.mapToInt(ele -> ele).toArray();
		return sortedIndices;
	}

	@Override
	public boolean isSorted() {
		for(int i = 1; i < cols.length; i++)
			if(cols[i - 1] > cols[i])
				return false;
		return true;
	}

	@Override
	public IColIndex sort() {
		int[] ret = new int[cols.length];
		System.arraycopy(cols, 0, ret, 0, cols.length);
		Arrays.sort(ret);
		return ColIndexFactory.create(ret);
	}

	@Override
	public boolean contains(int i) {
		if(i < cols[0] || i > cols[cols.length - 1])
			return false;
		int id = Arrays.binarySearch(cols, 0, cols.length, i);
		return id >= 0;
	}

	@Override
	public double avgOfIndex() {
		double s = 0.0;
		for(int i = 0; i < cols.length; i++) 
			s += cols[i];
		return s / cols.length;
	}

	protected class ArrayIterator implements IIterate {
		int id = 0;

		@Override
		public int next() {
			return cols[id++];
		}

		@Override
		public boolean hasNext() {
			return id < cols.length;
		}

		@Override
		public int v() {
			return cols[id];
		}

		@Override
		public int i() {
			return id;
		}
	}
}
