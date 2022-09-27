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

package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class OffsetSingle extends AOffset {
	private static final long serialVersionUID = -614636669776415032L;

	private final int off;

	public OffsetSingle(int off) {
		this.off = off;
	}

	@Override
	public AIterator getIterator() {
		return new IterateSingle();
	}

	@Override
	public AOffsetIterator getOffsetIterator() {
		return new IterateOffsetSingle();
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4;
	}

	@Override
	public int getSize() {
		return 1;
	}

	@Override
	public int getOffsetToFirst() {
		return off;
	}

	@Override
	public int getOffsetToLast() {
		return off;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize();
	}

	public static long estimateInMemorySize() {
		return 16 + 4; // object header plus int
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE_SPECIALIZATIONS.SINGLE_OFFSET.ordinal());
		out.writeInt(off);
	}

	public static OffsetSingle readFields(DataInput in) throws IOException {
		return new OffsetSingle(in.readInt());
	}

	@Override
	public OffsetSliceInfo slice(int l, int u) {

		if(l <= off && u > off)
			return new OffsetSliceInfo(0, 1, new OffsetSingle(off - l));
		else
			return new OffsetSliceInfo(-1, -1, new OffsetEmpty());

	}

	@Override
	protected AOffset moveIndex(int m) {
		return new OffsetSingle(off - m);
	}

	@Override
	protected int getLength() {
		return 1;
	}

	private class IterateSingle extends AIterator {

		private IterateSingle() {
			super(off);
		}

		@Override
		public int next() {
			return off;
		}

		@Override
		public int skipTo(int idx) {
			return off;
		}

		@Override
		public IterateSingle clone() {
			return this;
		}

		@Override
		public int getDataIndex() {
			return 0;
		}

		@Override
		public int getOffsetsIndex() {
			return 0;
		}
	}

	private class IterateOffsetSingle extends AOffsetIterator {

		private IterateOffsetSingle() {
			super(off);
		}

		@Override
		public int next() {
			return off;
		}
	}

}
