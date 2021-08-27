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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.utils.MemoryEstimates;

public class OffsetByte extends AOffset {

	private static final long serialVersionUID = -4716104973912491790L;

	private final static int maxV = 255;
	private final byte[] offsets;

	public OffsetByte(int[] indexes) {
		this(indexes, 0, indexes.length);
	}

	public OffsetByte(int[] indexes, int apos, int alen) {
		int endSize = 0;
		int ov = -1;
		for(int i = apos; i < alen; i++) {
			final int nv = indexes[i];
			endSize += 1 + (nv - ov) / maxV;
			ov = nv;
		}
		offsets = new byte[endSize];
		ov = -1;
		int p = 0;

		for(int i = apos; i < alen; i++) {
			final int nv = indexes[i];
			final int offsetSize = nv - ov;

			if(offsetSize == 0)
				throw new DMLCompressionException(
					"Invalid difference between cells index " + (i - 1) + " - " + (i) + "(" + ov + "," + nv + ")");
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

	}

	private OffsetByte(byte[] offsets) {
		this.offsets = offsets;
	}

	public byte[] getOffsets() {
		return offsets;
	}

	@Override
	public IterateByteOffset getIterator() {
		return new IterateByteOffset();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE.BYTE.ordinal());
		out.writeInt(offsets.length);
		for(byte o : offsets)
			out.writeByte(o);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(offsets.length);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + offsets.length;
	}

	@Override
	public int getSize() {
		int size = 0;
		for(byte b : offsets) {
			if(b != 0)
				size++;
		}
		return size;
	}

	public static long getInMemorySize(int length) {
		long size = 16 + 8; // object header plus reference
		size += MemoryEstimates.byteArrayCost(length);
		return size;
	}

	public static OffsetByte readFields(DataInput in) throws IOException {

		int offsetsLength = in.readInt();
		byte[] offsets = new byte[offsetsLength];
		for(int i = 0; i < offsetsLength; i++) {
			offsets[i] = in.readByte();
		}
		return new OffsetByte(offsets);
	}

	@Override
	public AIterator createIterator(int index, int dataIndex, int offset) {
		return new IterateByteOffset(index, dataIndex, offset);
	}

	public class IterateByteOffset extends AIterator {

		private IterateByteOffset() {
			super();
			byte v = offsets[index++];
			while(v == 0 && index < offsets.length) {
				offset += maxV;
				v = offsets[index++];
			}
			offset += v & 0xFF;
		}

		private IterateByteOffset(int index, int dataIndex, int offset) {
			super(index, dataIndex, offset);
		}

		@Override
		public void next() {
			byte v = offsets[index++];
			while(v == 0) {
				offset += maxV;
				if(index < offsets.length){
					v = offsets[index++];
				}
				else{
					break;
				}
			}
			offset += v & 0xFF;
			dataIndex++;
		}

		@Override
		public boolean hasNext() {
			return index < offsets.length;
		}

		@Override
		public IterateByteOffset clone() {
			return new IterateByteOffset(index, dataIndex, offset);
		}
	}
}
