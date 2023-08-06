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

import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public class SingleIndex extends AColIndex {
	private final int idx;

	public SingleIndex(int idx) {
		this.idx = idx;
	}

	@Override
	public int size() {
		return 1;
	}

	@Override
	public int get(int i) {
		return idx;
	}

	@Override
	public SingleIndex shift(int i) {
		return new SingleIndex(i + idx);
	}

	@Override
	public IIterate iterator() {
		return new SingleIterator();
	}

	public void write(DataOutput out) throws IOException {
		out.writeByte(ColIndexType.SINGLE.ordinal());
		out.writeInt(idx);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4;
	}

	@Override
	public long estimateInMemorySize() {
		return estimateInMemorySizeStatic();
	}

	public static long estimateInMemorySizeStatic() {
		return 16 + 4 + 4; // object, int, and padding
	}

	@Override
	public int findIndex(int i) {
		if(i < idx)
			return -1;
		else if(i == idx)
			return 0;
		else
			return -2;
	}

	@Override
	public SliceResult slice(int l, int u) {
		return (l <= idx && u > idx) //
			? l == 0 ? new SliceResult(0, 1, this) : new SliceResult(0, 1, new SingleIndex(idx - l)) //
			: new SliceResult(0, 0, null);
	}

	@Override
	public boolean equals(IColIndex other) {
		return other.size() == 1 && other.get(0) == idx;
	}

	@Override
	public IColIndex combine(IColIndex other) {
		if(other instanceof SingleIndex) {
			int otherV = other.get(0);
			if(otherV < idx)
				return new TwoIndex(otherV, idx);
			else
				return new TwoIndex(idx, otherV);
		}
		else
			return other.combine(this);
	}

	@Override
	public boolean isContiguous() {
		return true;
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
		return i == idx;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		sb.append(idx);
		sb.append("]");
		return sb.toString();
	}

	protected class SingleIterator implements IIterate {
		boolean taken = false;

		@Override
		public int next() {
			taken = true;
			return idx;
		}

		@Override
		public boolean hasNext() {
			return !taken;
		}

		@Override
		public int v() {
			return idx;
		}

		@Override
		public int i() {
			return 0;
		}
	}

}
