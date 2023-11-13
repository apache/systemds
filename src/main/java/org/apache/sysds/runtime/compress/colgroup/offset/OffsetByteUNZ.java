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
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class OffsetByteUNZ extends AOffsetByte {

	private static final long serialVersionUID = -4716104973912299990L;

	protected OffsetByteUNZ(byte[] offsets, int offsetToFirst, int offsetToLast) {
		super(offsets, offsetToFirst, offsetToLast);

		if(CompressedMatrixBlock.debug) {
			this.toString();
		}
	}

	@Override
	public AIterator getIterator() {
		return new IterateByteOffsetNoOverHalf();
	}

	@Override
	protected AIterator getIteratorFromIndexOff(int row, int dataIndex, int offIdx) {
		return new IterateByteOffsetNoOverHalf(dataIndex, row);
	}

	@Override
	public AOffsetIterator getOffsetIterator() {
		return new OffsetByteIteratorNoOverHalf();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		final byte[] its = new byte[4 * 3 + 1];
		its[0] = (byte) OffsetFactory.OFF_TYPE_SPECIALIZATIONS.BYTEUNZ.ordinal();
		IOUtilFunctions.intToBa(offsetToFirst, its, 1);
		IOUtilFunctions.intToBa(offsets.length, its, 5);
		IOUtilFunctions.intToBa(offsetToLast, its, 9);
		out.write(its);
		out.write(offsets);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + 4 + offsets.length;
	}

	@Override
	public int getSize() {
		return offsets.length + 1;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize(offsets.length);
	}

	public static long estimateInMemorySize(int nOffs) {
		long size = 16 + 4 + 4 + 8; // object header plus int plus reference
		size += MemoryEstimates.byteArrayCost(nOffs);
		return size;
	}

	public static AOffsetByte readFields(DataInput in) throws IOException {
		final int offsetToFirst = in.readInt();
		final int offsetsLength = in.readInt();
		final int offsetToLast = in.readInt();

		final byte[] offsets = new byte[offsetsLength];
		in.readFully(offsets);

		return new OffsetByteUNZ(offsets, offsetToFirst, offsetToLast);
	}

	@Override
	public OffsetSliceInfo slice(int lowOff, int highOff, int lowValue, int highValue, int low, int high) {
		byte[] newOffsets = Arrays.copyOfRange(offsets, lowOff, highOff);
		AOffset off = new OffsetByteUNZ(newOffsets, lowValue, highValue);
		return new OffsetSliceInfo(low, high + 1, off);
	}

	@Override
	public AOffset moveIndex(int m) {
		return new OffsetByteUNZ(offsets, offsetToFirst - m, offsetToLast - m);
	}

	private class IterateByteOffsetNoOverHalf extends AIterator {

		protected int index;

		private IterateByteOffsetNoOverHalf() {
			super(offsetToFirst);
		}

		private IterateByteOffsetNoOverHalf(int index, int offset) {
			super(offset);
			this.index = index;
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
		public int getDataIndex() {
			return index;
		}

		@Override
		public int getOffsetsIndex() {
			return index;
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

}
