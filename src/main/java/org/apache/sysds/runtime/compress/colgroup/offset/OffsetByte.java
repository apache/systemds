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

import org.apache.sysds.utils.MemoryEstimates;

public class OffsetByte extends AOffset {

	private static final long serialVersionUID = -4716104973912491790L;
	private static final int maxV = 255;

	private final byte[] offsets;
	private final int offsetToFirst;
	private final int offsetToLast;
	private final int size;
	private final boolean noOverHalf;
	private final boolean noZero;

	public OffsetByte(int[] indexes) {
		this(indexes, 0, indexes.length);
	}

	public OffsetByte(int[] indexes, int apos, int alen) {
		int endSize = 0;
		offsetToFirst = indexes[apos];
		offsetToLast = indexes[alen - 1];
		int ov = offsetToFirst;
		// find the size of the array
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			endSize += 1 + (nv - ov - 1) / maxV;
			ov = nv;
		}

		this.noZero = endSize == alen - apos - 1;
		offsets = new byte[endSize];
		ov = offsetToFirst;
		int p = 0;

		// populate the array
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			final int offsetSize = nv - ov;
			final int div = offsetSize / maxV;
			final int mod = offsetSize % maxV;
			if(mod == 0) {
				p += div - 1; // skip values
				offsets[p++] = (byte) maxV;
			}
			else {
				p += div; // skip values
				offsets[p++] = (byte) (mod);
			}

			ov = nv;
		}

		this.noOverHalf = getNoOverHalf();
		this.size = alen - apos;
	}

	protected OffsetByte(byte[] offsets, int offsetToFirst, int offsetToLast, int size) {
		this.offsets = offsets;
		this.offsetToFirst = offsetToFirst;
		this.offsetToLast = offsetToLast;
		this.noOverHalf = getNoOverHalf();
		this.noZero = getNoZero();
		this.size = size;
	}

	private boolean getNoOverHalf() {
		boolean noOverHalf = true;
		for(byte b : offsets)
			if(b < 1) {
				noOverHalf = false;
				break;
			}
		return noOverHalf;
	}

	private boolean getNoZero() {
		boolean noZero = true;
		for(byte b : offsets)
			if(b == 0) {
				noZero = false;
				break;
			}
		return noZero;
	}

	@Override
	public AIterator getIterator() {
		if(noOverHalf)
			return new IterateByteOffsetNoOverHalf();
		else if(noZero)
			return new IterateByteOffsetNoZero();
		else
			return new IterateByteOffset();
	}

	@Override
	public AOffsetIterator getOffsetIterator() {
		if(noOverHalf)
			return new OffsetByteIteratorNoOverHalf();
		else if(noZero)
			return new OffsetByteIteratorNoZero();
		else
			return new OffsetByteIterator();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE.BYTE.ordinal());
		out.writeInt(offsetToFirst);
		out.writeInt(offsets.length);
		out.writeInt(offsetToLast);
		out.writeInt(size);
		for(byte o : offsets)
			out.writeByte(o);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + 4 + 4 + offsets.length;
	}

	@Override
	public int getSize() {
		return size;
	}

	@Override
	public int getOffsetToFirst() {
		return offsetToFirst;
	}

	@Override
	public int getOffsetToLast() {
		return offsetToLast;
	}

	@Override
	public int getOffsetsLength() {
		return offsets.length;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize(offsets.length);
	}

	public static long estimateInMemorySize(int nOffs) {
		long size = 16 + 4 + 4 + 4 + 8; // object header plus int plus reference
		size += MemoryEstimates.byteArrayCost(nOffs);
		return size;
	}

	public static OffsetByte readFields(DataInput in) throws IOException {
		final int offsetToFirst = in.readInt();
		final int offsetsLength = in.readInt();
		final int offsetToLast = in.readInt();
		final int size = in.readInt();

		final byte[] offsets = new byte[offsetsLength];

		for(int i = 0; i < offsetsLength; i++)
			offsets[i] = in.readByte();

		return new OffsetByte(offsets, offsetToFirst, offsetToLast, size);
	}

	private class IterateByteOffset extends AIterator {

		protected int index;
		protected int dataIndex;

		private IterateByteOffset() {
			super(offsetToFirst);
			index = 0;
			dataIndex = 0;
		}

		private IterateByteOffset(int index, int dataIndex, int offset) {
			super(offset);
			this.index = index;
			this.dataIndex = dataIndex;
		}

		@Override
		public int next() {
			byte v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			offset += v & 0xFF;
			index++;
			dataIndex++;
			return offset;
		}

		@Override
		public int skipTo(int idx) {
			if(idx < offsetToLast)
				while(offset < idx)
					next();
			else
				while(offset < idx && index < offsets.length)
					next();
			return offset;
		}

		@Override
		public IterateByteOffset clone() {
			return new IterateByteOffset(index, dataIndex, offset);
		}

		@Override
		public int getDataIndex() {
			return dataIndex;
		}

		@Override
		public int getOffsetsIndex() {
			return index;
		}
	}

	private class IterateByteOffsetNoZero extends AIterator {

		protected int index;

		private IterateByteOffsetNoZero() {
			super(offsetToFirst);
		}

		private IterateByteOffsetNoZero(int index, int offset) {
			super(offset);
			this.index = index;
		}

		@Override
		public int next() {
			byte v = offsets[index];
			offset += v & 0xFF;
			index++;
			return offset;
		}

		@Override
		public int skipTo(int idx) {
			while(offset < idx && index < offsets.length)
				next();

			return offset;
		}

		@Override
		public IterateByteOffsetNoZero clone() {
			return new IterateByteOffsetNoZero(index, offset);
		}

		@Override
		public int getDataIndex() {
			return index;
		}

		@Override
		public int getOffsetsIndex() {
			return index;
		}
	}

	private class IterateByteOffsetNoOverHalf extends IterateByteOffsetNoZero {

		private IterateByteOffsetNoOverHalf() {
			super();
		}

		private IterateByteOffsetNoOverHalf(int index, int offset) {
			super(index, offset);
		}

		@Override
		public final int next() {
			offset += offsets[index];
			index++;
			return offset;
		}

		@Override
		public final int skipTo(int idx) {
			while(offset < idx && index < offsets.length) {
				offset += offsets[index];
				index++;
			}

			return offset;
		}

		@Override
		public final IterateByteOffsetNoOverHalf clone() {
			return new IterateByteOffsetNoOverHalf(index, offset);
		}
	}

	private class OffsetByteIteratorNoOverHalf extends AOffsetIterator {

		protected int index;

		private OffsetByteIteratorNoOverHalf() {
			super(offsetToFirst);
			index = 0;
		}

		@Override
		public int next() {
			return offset += offsets[index++];
		}
	}

	private class OffsetByteIterator extends AOffsetIterator {

		protected int index;

		private OffsetByteIterator() {
			super(offsetToFirst);
			index = 0;
		}

		@Override
		public int next() {
			byte v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			index++;
			return offset += v & 0xFF;
		}
	}

	private class OffsetByteIteratorNoZero extends AOffsetIterator {

		protected int index;

		private OffsetByteIteratorNoZero() {
			super(offsetToFirst);
			index = 0;
		}

		@Override
		public int next() {
			return offset += offsets[index++] & 0xFF;
		}
	}
}
