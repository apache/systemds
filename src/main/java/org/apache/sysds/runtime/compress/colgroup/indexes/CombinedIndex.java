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

public class CombinedIndex extends AColIndex {
	protected final IColIndex l;
	protected final IColIndex r;

	public CombinedIndex(IColIndex l, IColIndex r) {
		this.l = l;
		this.r = r;
	}

	@Override
	public int size() {
		return l.size() + r.size();
	}

	@Override
	public int get(int i) {
		if(i >= l.size())
			return r.get(i - l.size());
		else
			return l.get(i);
	}

	@Override
	public IColIndex shift(int i) {
		return new CombinedIndex(l.shift(i), r.shift(i));
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.write(ColIndexType.COMBINED.ordinal());
		l.write(out);
		r.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + l.getExactSizeOnDisk() + r.getExactSizeOnDisk();
	}

	@Override
	public long estimateInMemorySize() {
		return 16 + 8 + 8 + l.estimateInMemorySize() + r.estimateInMemorySize();
	}

	@Override
	public IIterate iterator() {
		return new CombinedIterator();
	}

	@Override
	public int findIndex(int i) {
		final int a = l.findIndex(i);
		if(a < 0) {
			final int b = r.findIndex(i);
			if(b < 0)
				return b + a + 1;
			else
				return b + l.size();
		}
		else
			return a;
	}

	@Override
	public SliceResult slice(int l, int u) {
		return getArrayIndex().slice(l, u);
	}

	@Override
	public boolean equals(IColIndex other) {
		if(other == this)
			return true;
		else if(size() == other.size()) {
			if(other instanceof CombinedIndex) {
				CombinedIndex o = (CombinedIndex) other;
				return o.l.equals(l) && o.r.equals(r);
			}
			else {
				IIterate t = iterator();
				IIterate o = other.iterator();

				while(t.hasNext()) {
					if(t.next() != o.next())
						return false;
				}
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

		final int[] ret = new int[sr + sl];
		IIterate t = iterator();
		IIterate o = other.iterator();
		int i = 0;
		while(t.hasNext() && o.hasNext()) {
			final int tv = t.v();
			final int ov = o.v();
			if(tv < ov) {
				ret[i++] = tv;
				t.next();
			}
			else {
				ret[i++] = ov;
				o.next();
			}
		}
		while(t.hasNext())
			ret[i++] = t.next();
		while(o.hasNext())
			ret[i++] = o.next();

		return ColIndexFactory.create(ret);

	}

	@Override
	public boolean isContiguous() {
		return false;
	}

	@Override
	public int[] getReorderingIndex() {
		return getArrayIndex().getReorderingIndex();
	}

	@Override
	public boolean isSorted() {
		return true;
	}

	@Override
	public IColIndex sort() {
		throw new DMLCompressionException("CombinedIndex is always sorted");
	}

	@Override
	public boolean contains(int i) {
		return l.contains(i) || r.contains(i);
	}

	@Override
	public double avgOfIndex() {
		double lv = l.avgOfIndex() * l.size();
		double rv = r.avgOfIndex() * r.size();
		return (lv + rv) / size();
	}

	private IColIndex getArrayIndex() {
		int s = size();
		int[] vals = new int[s];
		IIterate a = iterator();
		for(int i = 0; i < s; i++) {
			vals[i] = a.next();
		}
		return ColIndexFactory.create(vals);
	}

	public static CombinedIndex read(DataInput in) throws IOException {
		return new CombinedIndex(ColIndexFactory.read(in), ColIndexFactory.read(in));
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		sb.append(l);
		sb.append(", ");
		sb.append(r);
		sb.append("]");
		return sb.toString();
	}

	protected class CombinedIterator implements IIterate {
		boolean doneFirst = false;
		IIterate I = l.iterator();

		@Override
		public int next() {
			int v = I.next();
			if(!I.hasNext() && !doneFirst) {
				doneFirst = true;
				I = r.iterator();
			}
			return v;

		}

		@Override
		public boolean hasNext() {
			return I.hasNext() || doneFirst == false;
		}

		@Override
		public int v() {
			return I.v();
		}

		@Override
		public int i() {
			if(doneFirst) 
				return I.i() + l.size();
			else
				return I.i();
		}
	}

}
