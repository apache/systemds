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

import org.apache.sysds.runtime.compress.DMLCompressionException;

public class TwoRangesIndex extends AColIndex {

	/** The lower index range */
	protected final RangeIndex idx1;
	/** The upper index range */
	protected final RangeIndex idx2;

	public TwoRangesIndex(RangeIndex lower, RangeIndex higher) {
		this.idx1 = lower;
		this.idx2 = higher;
	}

	@Override
	public int size() {
		return idx1.size() + idx2.size();
	}

	@Override
	public int get(int i) {
		if(i < idx1.size())
			return idx1.get(i);
		else
			return idx2.get(i - idx1.size());
	}

	@Override
	public IColIndex shift(int i) {
		return new TwoRangesIndex(idx1.shift(i), idx2.shift(i));
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(ColIndexType.TWORANGE.ordinal());
		out.writeInt(idx1.get(0));
		out.writeInt(idx1.size());
		out.writeInt(idx2.get(0));
		out.writeInt(idx2.size());
	}

	public static TwoRangesIndex read(DataInput in) throws IOException {
		int l1 = in.readInt();
		int u1 = in.readInt() + l1;
		int l2 = in.readInt();
		int u2 = in.readInt() + l2;
		return new TwoRangesIndex(new RangeIndex(l1, u1), new RangeIndex(l2, u2));
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + 4 + 4;
	}

	@Override
	public long estimateInMemorySize() {
		return estimateInMemorySizeStatic();
	}

	public static long estimateInMemorySizeStatic() {
		return 16 + 8 + 8 + RangeIndex.estimateInMemorySizeStatic() * 2;
	}

	@Override
	public IIterate iterator() {
		return new TwoRangesIterator();
	}

	@Override
	public int findIndex(int i) {
		int aix = idx1.findIndex(i);
		if(aix < -1 * idx1.size()) {
			int bix = idx2.findIndex(i);
			if(bix < 0)
				return aix + bix + 1;
			else
				return idx1.size() + bix;
		}
		else
			return aix;

	}

	@Override
	public SliceResult slice(int l, int u) {
		if(u <= idx1.get(0))
			return new SliceResult(0, 0, null);
		else if(l >= idx2.get(idx2.size() - 1))
			return new SliceResult(0, 0, null);
		else if(l <= idx1.get(0) && u >= idx2.get(idx2.size() - 1)) {
			RangeIndex ids1 = idx1.shift(-l);
			RangeIndex ids2 = idx2.shift(-l);
			return new SliceResult(0, size(), new TwoRangesIndex(ids1, ids2));
		}

		SliceResult sa = idx1.slice(l, u);
		SliceResult sb = idx2.slice(l, u);
		if(sa.ret == null) {
			return new SliceResult(idx1.size() + sb.idStart, idx1.size() + sb.idEnd, sb.ret);
		}
		else if(sb.ret == null)
		// throw new NotImplementedException();
			return sa;
		else {
			IColIndex c = sa.ret.combine(sb.ret);
			return new SliceResult(sa.idStart, sa.idStart + sb.idEnd, c);
		}
	}

	@Override
	public boolean equals(IColIndex other) {
		if(other instanceof TwoRangesIndex) {
			TwoRangesIndex otri = (TwoRangesIndex) other;
			return idx1.equals(otri.idx1) && idx2.equals(otri.idx2);
		}
		else if(other instanceof RangeIndex)
			return false;
		else
			return other.equals(this);
	}

	@Override
	public IColIndex combine(IColIndex other) {
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
		return idx1.contains(i) || idx2.contains(i);
	}

	@Override
	public double avgOfIndex() {
		return (idx1.avgOfIndex() * idx1.size() + idx2.avgOfIndex() * idx2.size()) / size();
	}

	@Override
	public int hashCode() {
		// 811 is a prime.
		return idx1.hashCode() * 811 + idx2.hashCode();
	}

	@Override
	public boolean containsAny(IColIndex idx) {
		return idx1.containsAny(idx) || idx2.containsAny(idx);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		sb.append(idx1.get(0));
		sb.append(" -> ");
		sb.append(idx1.get(idx1.size()));
		sb.append(" And ");
		sb.append(idx2.get(0));
		sb.append(" -> ");
		sb.append(idx2.get(idx2.size()));
		sb.append("]");
		return sb.toString();
	}

	protected class TwoRangesIterator implements IIterate {
		IIterate a = idx1.iterator();
		IIterate b = idx2.iterator();
		boolean aDone = false;

		@Override
		public int next() {
			if(!aDone) {
				int v = a.next();
				aDone = !a.hasNext();
				return v;
			}
			else
				return b.next();
		}

		@Override
		public boolean hasNext() {
			return a.hasNext() || b.hasNext();
		}

		@Override
		public int v() {
			if(!aDone)
				return a.v();
			else
				return b.v();
		}

		@Override
		public int i() {
			if(!aDone)
				return a.i();
			else
				return a.i() + b.i();
		}
	}
}
