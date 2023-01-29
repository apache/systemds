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

public class RangeIndex implements IColIndex {
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
		return 16 + 8;
	}

	@Override
	public IIterate iterator() {
		return new RangeIterator();
	}

	protected static boolean isValidRange(int[] indexes) {
		int len = indexes.length;
		int first = indexes[0];
		int last = indexes[indexes.length - 1];
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
