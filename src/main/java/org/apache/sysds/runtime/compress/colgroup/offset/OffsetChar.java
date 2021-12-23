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

public class OffsetChar extends AOffset {

	private static final long serialVersionUID = -1192266421395964882L;
	private static final int maxV = (int) Character.MAX_VALUE;

	private final char[] offsets;
	private final int offsetToFirst;
	private final int offsetToLast;

	public OffsetChar(int[] indexes) {
		this(indexes, 0, indexes.length);
	}

	public OffsetChar(int[] indexes, int apos, int alen) {
		int endSize = 0;
		offsetToFirst = indexes[apos];
		offsetToLast = indexes[alen - 1];
		int ov = offsetToFirst;
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			endSize += 1 + (nv - ov - 1) / maxV;
			ov = nv;
		}
		offsets = new char[endSize];
		ov = offsetToFirst;
		int p = 0;

		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			final int offsetSize = (nv - ov);
			final int div = offsetSize / maxV;
			final int mod = offsetSize % maxV;
			if(mod == 0) {
				p += div - 1; // skip values
				offsets[p++] = (char) maxV;
			}
			else {
				p += div; // skip values
				offsets[p++] = (char) (mod);
			}

			ov = nv;
		}
	}

	private OffsetChar(char[] offsets, int offsetToFirst, int offsetToLast) {
		this.offsets = offsets;
		this.offsetToFirst = offsetToFirst;
		this.offsetToLast = offsetToLast;
	}

	@Override
	public IterateCharOffset getIterator() {
		return new IterateCharOffset();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE.CHAR.ordinal());
		out.writeInt(offsetToFirst);
		out.writeInt(offsets.length);
		for(char o : offsets)
			out.writeChar(o);
	}

	@Override
	public long getInMemorySize() {
		long size = 16 + 4 + 8; // object header plus int plus reference
		size += MemoryEstimates.charArrayCost(offsets.length);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + offsets.length * 2;
	}

	@Override
	public int getSize() {
		int size = 1;
		for(char b : offsets) {
			if(b != 0)
				size++;
		}
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

	public static OffsetChar readFields(DataInput in) throws IOException {
		final int offsetToFirst = in.readInt();
		final int offsetsLength = in.readInt();
		final char[] offsets = new char[offsetsLength];
		int offsetToLast = offsetToFirst;
		for(int i = 0; i < offsetsLength; i++) {
			offsets[i] = in.readChar();
			offsetToLast += offsets[i];
		}
		return new OffsetChar(offsets, offsetToFirst, offsetToLast);
	}

	public static long estimateInMemorySize(int nOffs, int nRows) {
		long size = 16 + 4 + 8; // object header plus int plus reference
		size += MemoryEstimates.charArrayCost(Math.max(nOffs, nRows / maxV));
		return size;
	}

	@Override
	protected final void preAggregateDenseMapRowByte(double[] mV, int off, double[] preAV, int cu, int nVal, byte[] data,
		AIterator it) {
		final int maxId = data.length - 1;
		while(it.isNotOver(cu)) {
			final int dx = it.getDataIndex();
			preAV[data[dx] & 0xFF] += mV[off + it.value()];
			if(dx < maxId)
				it.next();
			else
				break;
		}
		cacheIterator(it, cu);
	}

	@Override
	protected final void preAggregateDenseMapRowChar(double[] mV, int off, double[] preAV, int cu, int nVal, char[] data,
		AIterator it) {
		final int maxId = data.length - 1;
		while(it.isNotOver(cu)) {
			final int dx = it.getDataIndex();
			preAV[data[dx]] += mV[off + it.value()];
			if(dx < maxId)
				it.next();
			else
				break;
		}
		cacheIterator(it, cu);
	}

	@Override
	protected final void preAggregateDenseMapRowBit(double[] mV, int off, double[] preAV, int cu, int nVal, BitSet data,
		AIterator it) {
		int offset = it.offset + off;
		int index = it.index;
		int dataIndex = it.dataIndex;

		if(cu > offsetToLast) {
			final int last = offsetToLast + off;

			while(offset < last) {
				preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
				char v = offsets[index];
				while(v == 0) {
					offset += maxV;
					index++;
					v = offsets[index];
				}
				offset += v;
				index++;
				dataIndex++;
			}
			preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
		}
		else {
			final int last = cu + off;
			while(offset < last) {
				preAV[data.get(dataIndex) ? 1 : 0] += mV[offset];
				char v = offsets[index];
				while(v == 0) {
					offset += maxV;
					index++;
					v = offsets[index];
				}
				offset += v;
				index++;
				dataIndex++;
			}

		}
		it.offset = offset - off;
		it.dataIndex = index;
		it.index = index;
		cacheIterator(it, cu);
	}

	@Override
	protected void preAggregateDenseMapRowsByte(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		byte[] data, AIterator it) {

		final int offsetStart = it.offset;
		final int indexStart = it.index;
		final int dataIndexStart = it.dataIndex;
		if(cu < getOffsetToLast() + 1) {
			// inside offsets
			for(int r = rl; r < ru; r++) {
				final int offOut = (r - rl) * nVal;
				final double[] vals = db.values(r);
				final int off = db.pos(r);
				final int cur = cu + off;
				it.offset = offsetStart + off;
				it.index = indexStart;
				it.dataIndex = dataIndexStart;
				while(it.offset < cur) {
					preAV[offOut + data[it.dataIndex] & 0xFF] += vals[it.offset];
					it.next();
				}
				it.offset -= off;
			}
			cacheIterator(it, cu);
		}
		else {
			final int maxId = data.length - 1;
			// all the way to the end of offsets.
			for(int r = rl; r < ru; r++) {
				final int offOut = (r - rl) * nVal;
				final int off = db.pos(r);
				final double[] vals = db.values(r);
				it.offset = offsetStart + off;
				it.index = indexStart;
				it.dataIndex = dataIndexStart;
				preAV[offOut + data[it.dataIndex] & 0xFF] += vals[it.offset];
				while(it.dataIndex < maxId) {
					it.next();
					preAV[offOut + data[it.dataIndex] & 0xFF] += vals[it.offset];
				}
			}
		}
	}

	@Override
	protected void preAggregateDenseMapRowsChar(DenseBlock db, double[] preAV, int rl, int ru, int cl, int cu, int nVal,
		char[] data, AIterator it) {

		final int offsetStart = it.offset;
		final int indexStart = it.index;
		final int dataIndexStart = it.dataIndex;
		if(cu < getOffsetToLast() + 1) {

			for(int r = rl; r < ru; r++) {
				final int offOut = (r - rl) * nVal;
				final double[] vals = db.values(r);
				final int off = db.pos(r);
				final int cur = cu + off;
				it.offset = offsetStart + off;
				it.index = indexStart;
				it.dataIndex = dataIndexStart;
				while(it.offset < cur) {
					preAV[offOut + data[it.dataIndex]] += vals[it.offset];
					it.next();
				}
				it.offset -= off;
			}

			cacheIterator(it, cu);
		}
		else {
			final int maxId = data.length - 1;
			// all the way to the end.
			for(int r = rl; r < ru; r++) {
				final int offOut = (r - rl) * nVal;
				final int off = db.pos(r);
				final double[] vals = db.values(r);
				it.offset = offsetStart + off;
				it.index = indexStart;
				it.dataIndex = dataIndexStart;
				preAV[offOut + data[it.dataIndex]] += vals[it.offset];
				while(it.dataIndex < maxId) {
					it.next();
					preAV[offOut + data[it.dataIndex]] += vals[it.offset];
				}
			}
		}
	}

	private class IterateCharOffset extends AIterator {

		private IterateCharOffset() {
			super(0, 0, offsetToFirst);
		}

		private IterateCharOffset(int index, int dataIndex, int offset) {
			super(index, dataIndex, offset);
		}

		@Override
		public void next() {
			char v = offsets[index];
			while(v == 0) {
				offset += maxV;
				index++;
				v = offsets[index];
			}
			offset += v;
			index++;
			dataIndex++;
		}

		@Override
		public int value() {
			return offset;
		}

		@Override
		public int skipTo(int idx) {
			while(offset < idx && index < offsets.length)
				next();
			return offset;
		}

		@Override
		public IterateCharOffset clone() {
			return new IterateCharOffset(index, dataIndex, offset);
		}
	}
}
