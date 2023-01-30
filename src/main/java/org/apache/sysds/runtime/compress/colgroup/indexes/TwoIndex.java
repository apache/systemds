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

import org.apache.commons.lang.NotImplementedException;

public class TwoIndex implements IColIndex {
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
		return 16 + 8; // object, 2x int
	}

	@Override
	public int findIndex(int i) {
		return i == id1 ? 0 : i == id2 ? 1 : -1;
	}

	@Override
	public SliceResult slice(int l, int u) {
		throw new NotImplementedException();
	}

	@Override
	public boolean equals(Object other) {
		throw new NotImplementedException();
	}

	@Override
	public boolean equals(IColIndex other) {
		throw new NotImplementedException();
	}

	@Override
	public int hashCode() {
		throw new NotImplementedException();
	}

	@Override
	public boolean contains(IColIndex a, IColIndex b) {
		throw new NotImplementedException();
	}

	@Override
	public IColIndex combine(IColIndex other) {
		throw new NotImplementedException();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" [");
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
