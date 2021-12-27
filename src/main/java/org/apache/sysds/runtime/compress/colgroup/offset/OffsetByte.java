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
import java.util.BitSet;

import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class OffsetByte extends AOffset {

	private static final long serialVersionUID = -4716104973912491790L;
	private static final int maxV = 255;

	private final byte[] offsets;
	private final int offsetToFirst;
	private final int offsetToLast;
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
		this.noZero = getNoZero();
	}

	protected OffsetByte(byte[] offsets, int offsetToFirst, int offsetToLast) {
		this.offsets = offsets;
		this.offsetToFirst = offsetToFirst;
		this.offsetToLast = offsetToLast;
		this.noOverHalf = getNoOverHalf();
		this.noZero = getNoZero();
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
	public IterateByteOffset getIterator() {
		if(noOverHalf)
			return new IterateByteOffsetNoOverHalf();
		else if(noZero)
			return new IterateByteOffsetNoZero();
		else
			return new IterateByteOffset();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE.BYTE.ordinal());
		out.writeInt(offsetToFirst);
		out.writeInt(offsets.length);
		for(byte o : offsets)
			out.writeByte(o);
	}

	@Override
	public long getInMemorySize() {
		long size = 16 + 4 + 4 + 8; // object header plus ints plus reference
		size += MemoryEstimates.byteArrayCost(offsets.length);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + offsets.length;
	}

	@Override
	public int getSize() {
		int size = 1;
		for(byte b : offsets)
			if(b != 0)
				size++;

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

	public static long estimateInMemorySize(int nOffs, int nRows) {
		long size = 16 + 4 + 4 + 8; // object header plus int plus reference
		size += MemoryEstimates.byteArrayCost(Math.max(nOffs, nRows / maxV));
		return size;
	}

	public static OffsetByte readFields(DataInput in) throws IOException {
		final int offsetToFirst = in.readInt();
		final int offsetsLength = in.readInt();

		final byte[] offsets = new byte[offsetsLength];
		int offsetToLast = offsetToFirst;
		for(int i = 0; i < offsetsLength; i++) {
			offsets[i] = in.readByte();
			offsetToLast += offsets[i] & 0xFF;
		}
		return new OffsetByte(offsets, offsetToFirst, offsetToLast);
	}

	@Override
	protected final void preAggregateDenseMapRowByte(double[] mV, int off, double[] preAV, int cu, int nVal, byte[] data,
		AIterator it) {
		IterateByteOffset itb = (IterateByteOffset) it;
		if(cu < offsetToLast + 1) {
			final boolean nvalHalf = nVal < 127;
			if(noOverHalf && noZero && nvalHalf)
				preAggregateDenseByteMapRowBelowEndAndNoZeroNoOverHalfAlsoData(mV, off, preAV, cu, data, itb);
			else if(noOverHalf && noZero)
				preAggregateDenseByteMapRowBelowEndAndNoZeroNoOverHalf(mV, off, preAV, cu, data, itb);
			else if(noZero)
				preAggregateDenseByteMapRowBelowEndAndNoZero(mV, off, preAV, cu, data, itb);
			else if(nvalHalf)
				preAggregateDenseByteMapRowBelowEndDataHalf(mV, off, preAV, cu, data, itb);
			else if(noOverHalf)
				preAggregateDenseByteMapRowBelowEndNoOverHalf(mV, off, preAV, cu, data, itb);
			else
				preAggregateDenseByteMapRowBelowEnd(mV, off, preAV, cu, data, itb);
			cacheIterator(itb, cu);
		}
		else if(noZero)
			preAggregateDenseByteMapRowNoZero(mV, off, preAV, data, itb);
		else
			preAggregateDenseByteMapRow(mV, off, preAV, data, itb);

	}

	private final void preAggregateDenseByteMapRow(double[] mV, int off, double[] preAV, byte[] data,
		IterateByteOffset it) {
		final int maxId = data.length - 1;

		int offset = it.offset + off;
		int index = it.getOffsetsIndex();
		int dataIndex = it.getDataIndex();

		preAV[data[dataIndex] & 0xFF] += mV[offset];
		while(dataIndex < maxId) {
			byte v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			offset += v & 0xFF;
			index++;
			dataIndex++;
			preAV[data[dataIndex] & 0xFF] += mV[offset];
		}
	}

	private final void preAggregateDenseByteMapRowNoZero(double[] mV, int off, double[] preAV, byte[] data,
		IterateByteOffset it) {

		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		while(index < offsets.length) {
			preAV[data[index] & 0xFF] += mV[offset];
			offset += offsets[index++] & 0xFF;
		}
		// process straggler index.
		preAV[data[index] & 0xFF] += mV[offset];
	}

	private void preAggregateDenseByteMapRowBelowEndNoOverHalf(double[] mV, int off, double[] preAV, int cu, byte[] data,
		IterateByteOffset it) {

		cu += off;
		it.offset += off;
		while(it.offset < cu) {
			preAV[data[it.getDataIndex()] & 0xFF] += mV[it.offset];
			byte v = offsets[it.index];
			while(v == 0) {
				it.offset += maxV;
				it.index++;
				v = offsets[it.index];
			}
			it.offset += v;
			it.index++;
			it.dataIndex++;
		}
		it.offset -= off;
	}

	private void preAggregateDenseByteMapRowBelowEndDataHalf(double[] mV, int off, double[] preAV, int cu, byte[] data,
		IterateByteOffset it) {

		cu += off;
		it.offset += off;
		while(it.offset < cu) {
			preAV[data[it.getDataIndex()]] += mV[it.offset];
			byte v = offsets[it.index];
			while(v == 0) {
				it.offset += maxV;
				it.index++;
				v = offsets[it.index];
			}
			it.offset += v & 0xFF;
			it.index++;
			it.dataIndex++;
		}
		it.offset -= off;
	}

	private void preAggregateDenseByteMapRowBelowEnd(double[] mV, int off, double[] preAV, int cu, byte[] data,
		IterateByteOffset it) {

		cu += off;
		it.offset += off;
		while(it.offset < cu) {
			preAV[data[it.getDataIndex()] & 0xFF] += mV[it.offset];
			byte v = offsets[it.index];
			while(v == 0) {
				it.offset += maxV;
				it.index++;
				v = offsets[it.index];
			}
			it.offset += v & 0xFF;
			it.index++;
			it.dataIndex++;
		}
		it.offset -= off;
	}

	private void preAggregateDenseByteMapRowBelowEndAndNoZero(double[] mV, int off, double[] preAV, int cu, byte[] data,
		IterateByteOffset it) {

		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		cu += off;

		while(offset < cu) {
			preAV[data[index] & 0xFF] += mV[offset];
			offset += offsets[index++] & 0xFF;
		}

		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
	}

	private final void preAggregateDenseByteMapRowBelowEndAndNoZeroNoOverHalf(double[] mV, int off, double[] preAV,
		int cu, byte[] data, IterateByteOffset it) {
		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		cu += off;

		while(offset < cu) {
			preAV[data[index] & 0xFF] += mV[offset];
			offset += offsets[index++];
		}

		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
	}

	private final void preAggregateDenseByteMapRowBelowEndAndNoZeroNoOverHalfAlsoData(double[] mV, int off,
		double[] preAV, int cu, byte[] data, IterateByteOffset it) {
		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		cu += off;

		while(offset < cu) {
			preAV[data[index]] += mV[offset];
			offset += offsets[index++];
		}

		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
	}

	@Override
	protected final void preAggregateDenseMapRowChar(double[] mV, int off, double[] preAV, int cu, int nVal, char[] data,
		AIterator it) {
		IterateByteOffset itb = (IterateByteOffset) it;
		final boolean noZero = offsets.length == data.length - 1;
		if(cu < offsetToLast + 1) {
			if(noOverHalf && noZero)
				preAggregateDenseCharMapRowBelowEndAndNoZeroNoOverHalf(mV, off, preAV, cu, data, itb);
			else if(noZero)
				preAggregateDenseCharMapRowBelowEndAndNoZero(mV, off, preAV, cu, data, itb);
			else
				preAggregateDenseCharMapRowBelowEnd(mV, off, preAV, cu, data, itb);
			cacheIterator(itb, cu);
		}
		else if(noZero)
			preAggregateDenseCharMapRowNoZero(mV, off, preAV, data, itb);
		else
			preAggregateDenseCharMapRow(mV, off, preAV, data, itb);
	}

	private void preAggregateDenseCharMapRow(double[] mV, int off, double[] preAV, char[] data, IterateByteOffset it) {
		final int maxId = data.length - 1;
		int offset = it.offset + off;
		int index = it.getOffsetsIndex();
		int dataIndex = it.getDataIndex();

		preAV[data[dataIndex]] += mV[offset];
		while(dataIndex < maxId) {
			byte v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			offset += v & 0xff;
			index++;
			dataIndex++;
			preAV[data[dataIndex]] += mV[offset];
		}
	}

	private void preAggregateDenseCharMapRowNoZero(double[] mV, int off, double[] preAV, char[] data,
		IterateByteOffset it) {

		int offset = it.offset + off;
		int index = it.getOffsetsIndex();
		while(index < offsets.length) {
			preAV[data[index]] += mV[offset];
			offset += offsets[index++] & 0xFF;
		}
		preAV[data[index]] += mV[offset];
	}

	private void preAggregateDenseCharMapRowBelowEnd(double[] mV, int off, double[] preAV, int cu, char[] data,
		IterateByteOffset it) {

		cu += off;
		it.offset += off;
		while(it.offset < cu) {
			preAV[data[it.getDataIndex()]] += mV[it.offset];
			byte v = offsets[it.index];
			while(v == 0) {
				it.offset += maxV;
				it.index++;
				v = offsets[it.index];
			}
			it.offset += v & 0xFF;
			it.index++;
			it.dataIndex++;
		}
		it.offset -= off;
	}

	private void preAggregateDenseCharMapRowBelowEndAndNoZero(double[] mV, int off, double[] preAV, int cu, char[] data,
		IterateByteOffset it) {
		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		cu += off;

		while(offset < cu) {
			preAV[data[index]] += mV[offset];
			offset += offsets[index++] & 0xFF;
		}

		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
	}

	private final void preAggregateDenseCharMapRowBelowEndAndNoZeroNoOverHalf(double[] mV, int off, double[] preAV,
		int cu, char[] data, IterateByteOffset it) {
		int offset = it.offset + off;
		int index = it.getOffsetsIndex();

		cu += off;

		while(offset < cu) {
			preAV[data[index]] += mV[offset];
			offset += offsets[index++];
		}

		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
	}

	@Override
	protected final void preAggregateDenseMapRowBit(double[] mV, int off, double[] preAV, int cu, int nVal, BitSet data,
		AIterator it) {
		IterateByteOffset itb = (IterateByteOffset) it;
		int offset = itb.offset + off;
		int index = itb.getOffsetsIndex();
		int dataIndex = itb.getDataIndex();

		if(cu > offsetToLast) {
			final int last = offsetToLast + off;

			while(offset < last) {
				preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
				byte v = offsets[index];
				while(v == 0) {
					offset += maxV;
					index++;
					v = offsets[index];
				}
				offset += v & 0xFF;
				index++;
				dataIndex++;
			}
			preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
		}
		else {
			final int last = cu + off;
			while(offset < last) {
				preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
				byte v = offsets[index];
				while(v == 0) {
					offset += maxV;
					index++;
					v = offsets[index];
				}
				offset += v & 0xFF;
				index++;
				dataIndex++;
			}

		}
		itb.offset = offset - off;
		itb.dataIndex = index;
		itb.index = index;
		cacheIterator(it, cu);
	}

	@Override
	protected void preAggregateDenseMapRowsByte(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		byte[] data, AIterator it) {
		IterateByteOffset itb = (IterateByteOffset) it;
		if(cu < getOffsetToLast() + 1)
			preAggregateDenseMapRowsByteBelowEnd(db, preAV, rl, ru, cl, cu, nVal, data, itb);
		else
			preAggregateDenseMapRowsByteEnd(db, preAV, rl, ru, cl, cu, nVal, data, itb);
	}

	private void preAggregateDenseMapRowsByteBelowEnd(DenseBlock db, final double[] preAV, final int rl, final int ru,
		final int cl, final int cu, final int nVal, byte[] data, IterateByteOffset it) {
		final double[] vals = db.values(rl);
		final int nCol = db.getCumODims(0);
		while(it.offset < cu) {
			final int dataOffset = data[it.getDataIndex()] & 0xFF;
			final int start = it.offset + nCol * rl;
			final int end = it.offset + nCol * ru;
			for(int offOut = dataOffset, off = start; off < end; offOut += nVal, off += nCol)
				preAV[offOut] += vals[off];
			it.next();
		}

		cacheIterator(it, cu);
	}

	private void preAggregateDenseMapRowsByteEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		byte[] data, IterateByteOffset it) {
		final int maxId = data.length - 1;
		final int offsetStart = it.offset;
		final int indexStart = it.getOffsetsIndex();
		final int dataIndexStart = it.getDataIndex();
		// all the way to the end of offsets.
		for(int r = rl; r < ru; r++) {
			final int offOut = (r - rl) * nVal;
			final int off = db.pos(r);
			final double[] vals = db.values(r);
			it.offset = offsetStart + off;
			it.index = indexStart;
			it.dataIndex = dataIndexStart;
			preAV[offOut + data[it.getDataIndex()] & 0xFF] += vals[it.offset];
			while(it.getDataIndex() < maxId) {
				it.next();
				preAV[offOut + data[it.getDataIndex()] & 0xFF] += vals[it.offset];
			}
		}
	}

	@Override
	protected void preAggregateDenseMapRowsChar(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		char[] data, AIterator it) {
		IterateByteOffset itb = (IterateByteOffset) it;
		if(cu < getOffsetToLast() + 1)
			preAggregateDenseMapRowsCharBelowEnd(db, preAV, rl, ru, cl, cu, nVal, data, itb);
		else
			preAggregateDenseMapRowsCharEnd(db, preAV, rl, ru, cl, cu, nVal, data, itb);

	}

	private void preAggregateDenseMapRowsCharBelowEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu,
		int nVal, char[] data, IterateByteOffset it) {
		final double[] vals = db.values(rl);
		while(it.offset < cu) {
			final int dataOffset = data[it.getDataIndex()];
			for(int r = rl, offOut = dataOffset; r < ru; r++, offOut += nVal)
				preAV[offOut] += vals[it.offset + db.pos(r)];
			it.next();
		}
		cacheIterator(it, cu);
	}

	private void preAggregateDenseMapRowsCharEnd(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		char[] data, IterateByteOffset it) {
		final int maxId = data.length - 1;
		// all the way to the end.
		final int offsetStart = it.offset;
		final int indexStart = it.getOffsetsIndex();
		final int dataIndexStart = it.getDataIndex();
		for(int r = rl; r < ru; r++) {
			final int offOut = (r - rl) * nVal;
			final int off = db.pos(r);
			final double[] vals = db.values(r);
			it.offset = offsetStart + off;
			it.index = indexStart;
			it.dataIndex = dataIndexStart;
			preAV[offOut + data[it.getDataIndex()]] += vals[it.offset];
			while(it.getDataIndex() < maxId) {
				it.next();
				preAV[offOut + data[it.getDataIndex()]] += vals[it.offset];
			}
		}
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
		public void next() {
			byte v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			offset += v & 0xFF;
			index++;
			dataIndex++;
		}

		@Override
		public int skipTo(int idx) {
			if(noOverHalf) {
				while(offset < idx && index < offsets.length) {
					offset += offsets[index];
					index++;
				}
				dataIndex = index;
			}
			else if(idx < offsetToLast)
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

	private class IterateByteOffsetNoZero extends IterateByteOffset {

		private IterateByteOffsetNoZero() {
			super();
		}

		private IterateByteOffsetNoZero(int index, int dataIndex, int offset) {
			super(index, dataIndex, offset);
		}

		@Override
		public void next() {
			byte v = offsets[index];
			offset += v & 0xFF;
			index++;
			dataIndex++;
		}

		@Override
		public int skipTo(int idx) {
			while(offset < idx && index < offsets.length) {
				int v = offsets[index] & 0xFF;
				offset += v;
				index++;
			}
			dataIndex = index;

			return offset;
		}

		@Override
		public IterateByteOffsetNoZero clone() {
			return new IterateByteOffsetNoZero(index, dataIndex, offset);
		}

	}

	private class IterateByteOffsetNoOverHalf extends IterateByteOffset {

		private IterateByteOffsetNoOverHalf() {
			super();
		}

		private IterateByteOffsetNoOverHalf(int index, int dataIndex, int offset) {
			super(index, dataIndex, offset);
		}

		@Override
		public void next() {
			offset += offsets[index];
			index++;
			dataIndex++;
		}

		@Override
		public int skipTo(int idx) {
			while(offset < idx && index < offsets.length) {
				offset += offsets[index];
				index++;
			}
			dataIndex = index;

			return offset;
		}

		@Override
		public IterateByteOffsetNoOverHalf clone() {
			return new IterateByteOffsetNoOverHalf(index, dataIndex, offset);
		}
	}
}
