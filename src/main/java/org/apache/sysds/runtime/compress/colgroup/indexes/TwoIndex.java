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
import java.util.Arrays;

public class TwoIndex extends AColIndex {
	private final int id1;
	private final int id2;

	public TwoIndex(int id1, int id2) {
		this.id1 = id1;
		this.id2 = id2;
	}

	@Override
	public int size() {
		return 2;
	}

	@Override
	public int get(int i) {
		if(i == 0)
			return id1;
		else
			return id2;
	}

	@Override
	public TwoIndex shift(int i) {
		return new TwoIndex(id1 + i, id2 + i);
	}

	@Override
	public IIterate iterator() {
		return new TwoIterator();
	}

	public void write(DataOutput out) throws IOException {
		out.writeByte(ColIndexType.TWO.ordinal());
		out.writeInt(id1);
		out.writeInt(id2);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4;
	}

	@Override
	public long estimateInMemorySize() {
		return estimateInMemorySizeStatic(); // object, 2x int
	}

	public static long estimateInMemorySizeStatic() {
		return 16 + 8;
	}

	@Override
	public int findIndex(int i) {
		if(i < id1)
			return -1;
		else if(i == id1)
			return 0;
		else if(i < id2)
			return -2;
		else if(i == id2)
			return 1;
		else
			return -3;
	}

	@Override
	public SliceResult slice(int l, int u) {
		SliceResult ret;
		if(l <= id1 && u > id2)
			ret = new SliceResult(0, 2, l == 0 ? this : new TwoIndex(id1 - l, id2 - l));
		else if(l <= id1 && u > id1)
			ret = new SliceResult(0, 1, new SingleIndex(id1 - l));
		else if(l <= id2 && u > id2)
			ret = new SliceResult(1, 2, new SingleIndex(id2 - l));
		else
			ret = new SliceResult(0, 0, null);
		return ret;
	}

	@Override
	public boolean equals(IColIndex other) {
		return other.size() == 2 && other.get(0) == id1 && other.get(1) == id2;
	}

	@Override
	public IColIndex combine(IColIndex other) {
		if(other instanceof SingleIndex) {
			int otherV = other.get(0);
			if(otherV < id1)
				return new ArrayIndex(new int[] {otherV, id1, id2});
			else if(otherV < id2)
				return new ArrayIndex(new int[] {id1, otherV, id2});
			else
				return new ArrayIndex(new int[] {id1, id2, otherV});
		}
		else if(other instanceof TwoIndex) {
			int[] vals = new int[] {other.get(0), other.get(1), id1, id2};
			Arrays.sort(vals);
			return new ArrayIndex(vals);
		}
		else
			return other.combine(this);
	}

	@Override
	public boolean isContiguous() {
		return id1 + 1 == id2;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("[");
		sb.append(id1);
		sb.append(", ");
		sb.append(id2);
		sb.append("]");
		return sb.toString();
	}

	protected class TwoIterator implements IIterate {
		int id = 0;

		@Override
		public int next() {
			if(id++ == 0)
				return id1;
			else
				return id2;
		}

		@Override
		public boolean hasNext() {
			return id < 2;
		}
	}

}
