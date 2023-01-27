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

import org.apache.sysds.utils.MemoryEstimates;

public class ArrayIndex implements IColIndex {
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
		return 16 + (long) MemoryEstimates.intArrayCost(cols.length);
	}

	@Override
	public IIterate iterator() {
		return new ArrayIterator();
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
	}
}
