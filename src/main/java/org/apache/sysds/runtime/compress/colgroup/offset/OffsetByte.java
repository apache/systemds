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
import java.util.Arrays;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AOffsetsGroup;
import org.apache.sysds.utils.MemoryEstimates;

public class OffsetByte extends AOffsetByte {

	private static final long serialVersionUID = -4716104973912491790L;
	protected static final int maxV = 255;

	private final int size;

	protected OffsetByte(byte[] offsets, int offsetToFirst, int offsetToLast, int size) {
		super(offsets, offsetToFirst, offsetToLast);
		this.size = size;
		if(CompressedMatrixBlock.debug) {
			this.toString();
		}
	}

	protected static AOffsetByte create(byte[] offsets, int offsetToFirst, int offsetToLast, int size, boolean noZero,
		boolean ub) {
		if(noZero) {
			if(ub)
				return new OffsetByteUNZ(offsets, offsetToFirst, offsetToLast);
			else
				return new OffsetByteNZ(offsets, offsetToFirst, offsetToLast);
		}
		else
			return new OffsetByte(offsets, offsetToFirst, offsetToLast, size);
	}

	@Override
	public AIterator getIterator() {
		return new IterateByteOffset();
	}

	@Override
	protected AIterator getIteratorFromIndexOff(int row, int dataIndex, int offIdx) {
		return new IterateByteOffset(offIdx, dataIndex, row);
	}

	@Override
	public AOffsetIterator getOffsetIterator() {
		return new OffsetByteIterator();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE_SPECIALIZATIONS.BYTE.ordinal());
		out.writeInt(offsetToFirst);
		out.writeInt(offsets.length);
		out.writeInt(offsetToLast);
		out.writeInt(size);
		out.write(offsets);
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
		in.readFully(offsets);

		return new OffsetByte(offsets, offsetToFirst, offsetToLast, size);
	}

	@Override
	public OffsetSliceInfo slice(int lowOff, int highOff, int lowValue, int highValue, int low, int high) {
		int newSize = high - low + 1;
		byte[] newOffsets = Arrays.copyOfRange(offsets, lowOff, highOff);
		AOffset off = new OffsetByte(newOffsets, lowValue, highValue, newSize);
		return new OffsetSliceInfo(low, high + 1, off);
	}

	@Override
	public AOffset moveIndex(int m) {
		return new OffsetByte(offsets, offsetToFirst - m, offsetToLast - m, size);
	}

	@Override
	public final AOffset appendN(AOffsetsGroup[] g, int s) {

		for(AOffsetsGroup gs : g) {
			final AOffset a = gs.getOffsets();
			if(!(a instanceof OffsetByte))
				return super.appendN(g, s);
		}

		// calculate byte array size.
		int totalLength = g[0].getOffsets().getLength();
		for(int i = 1; i < g.length; i++) {
			totalLength += g[i].getOffsets().getLength() + 1;
			int remainder = s - g[i - 1].getOffsets().getOffsetToLast();
			totalLength += (remainder + g[i].getOffsets().getOffsetToFirst() - 1) / maxV;
		}

		final byte[] ret = new byte[totalLength];

		int p = 0;
		int remainderLast = 0;
		int size = 0;
		boolean first = true;
		for(AOffsetsGroup gs : g) {

			final OffsetByte b = (OffsetByte) gs.getOffsets();
			if(!first) {
				final int offFirst = remainderLast + b.offsetToFirst;
				final int div = offFirst / OffsetByte.maxV;
				final int mod = offFirst % OffsetByte.maxV;
				if(mod == 0) {
					p += div - 1; // skip values
					ret[p++] = (byte) OffsetByte.maxV;
				}
				else {
					p += div; // skip values
					ret[p++] = (byte) (mod);
				}
			}

			final byte[] bd = b.offsets;
			System.arraycopy(bd, 0, ret, p, bd.length);
			remainderLast = s - b.offsetToLast;
			size += b.size;
			p += bd.length;
			first = false;
		}

		final int offLast = s * (g.length - 1) + g[g.length - 1].getOffsets().getOffsetToLast();
		return new OffsetByte(ret, offsetToFirst, offLast, size);
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

}
