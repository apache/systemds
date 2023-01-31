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

import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class RangeIndex extends AColIndex {
	private final int l;
	private final int u; // not inclusive

	public RangeIndex(int nCol) {
		l = 0;
		u = nCol;
	}

	public RangeIndex(int l, int u) {
		this.l = l;
		this.u = u;
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
	public IColIndex shift(int i) {
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
			int offL = Math.max(l, this.l) - this.l;
			int offR = Math.min(u, this.u) - this.l;
			return new SliceResult(offL, offR, new RangeIndex(Math.max(l, this.l) - l, Math.min(u, this.u) - l));
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
	public boolean isContiguous(){
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" [");
		sb.append(l);
		sb.append(" -> ");
		sb.append(u);
		sb.append("]");
		return sb.toString();
	}

	protected static boolean isValidRange(int[] indexes) {
		int len = indexes.length;
		int first = indexes[0];
		int last = indexes[indexes.length - 1];
		return last - first + 1 == len;
	}

	protected static boolean isValidRange(IntArrayList indexes) {
		int len = indexes.size();
		int first = indexes.get(0);
		int last = indexes.get(indexes.size() - 1);
		return last - first + 1 == len;
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
	}
}
