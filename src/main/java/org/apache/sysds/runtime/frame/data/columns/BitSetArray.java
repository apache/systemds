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

package org.apache.sysds.runtime.frame.data.columns;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.utils.MemoryEstimates;

public class BitSetArray extends Array<Boolean> {

	private static boolean useVectorizedKernel = true;
	private BitSet _data;

	protected BitSetArray(int size) {
		_size = size;
		_data = new BitSet(size);
	}

	public BitSetArray(boolean[] data) {
		_size = data.length;
		_data = new BitSet(data.length);
		// set bits.
		for(int i = 0; i < data.length; i++)
			if(data[i]) // slightly more efficient to check.
				_data.set(i);
	}

	public BitSetArray(BitSet data, int size) {
		_size = size;
		_data = data;
	}

	public BitSet get() {
		return _data;
	}

	@Override
	public Boolean get(int index) {
		return _data.get(index);
	}

	@Override
	public void set(int index, Boolean value) {
		_data.set(index, value != null ? value : false);
	}

	@Override
	public void set(int index, double value) {
		_data.set(index, value == 0 ? false : true);
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		throw new NotImplementedException();
	}

	private static long[] toLongArrayPadded(BitSet data, int minLength) {
		long[] ret = data.toLongArray();
		final int len = minLength / 64 + 1;
		if(ret.length != len) // make sure ret have allocated enough longs
			return Arrays.copyOf(ret, len);
		return ret;
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value, int rlSrc) {
		if(useVectorizedKernel && value instanceof BitSetArray && (ru - rl >= 64))
			setVectorized(rl, ru, (BitSetArray) value, rlSrc);
		else // default
			for(int i = rl, off = rlSrc; i <= ru; i++, off++)
				_data.set(i, value.get(off));
	}

	private void setVectorized(int rl, int ru, BitSetArray value, int rlSrc) {
		final int rangeLength = ru - rl + 1;
		final long[] otherValues = toLongArrayPadded(//
			(BitSet) value.get().get(rlSrc, rangeLength + rlSrc), rangeLength);
		long[] ret = toLongArrayPadded(_data, size());

		ret = setVectorizedLongs(rl, ru, otherValues, ret);
		_data = BitSet.valueOf(ret);
	}

	private static long[] setVectorizedLongs(int rl, int ru, long[] ov, long[] ret) {
		final long remainder = rl % 64L;
		if(remainder == 0)
			return setVectorizedLongsNoOffset(rl, ru, ov, ret);
		else 
			return setVectorizedLongsWithOffset(rl, ru, ov, ret);
	}

	private static long[] setVectorizedLongsNoOffset(int rl, int ru, long[] ov, long[] ret) {
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;
		int retP = rl / 64;

		// assign all full.
		for(int j = 0; j < ov.length - 1; j++) {
			ret[retP] = ov[j];
			retP++;
		}

		// handle tail.
		if(remainderEnd != 0) {
			// clear ret in the area.
			final long r = (ret[retP] >>> remainderEnd) << remainderEnd;
			final long v = (ov[ov.length - 1] << remainderEndInv) >>> remainderEndInv;
			// assign ret in the area.
			ret[retP] = r ^ v;
		}
		else
			ret[retP] = ov[ov.length - 1];
		return ret;
	}

	private static long[] setVectorizedLongsWithOffset(int rl, int ru, long[] ov, long[] ret) {
		final long remainder = rl % 64L;
		final long invRemainder = 64L - remainder;
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;

		int retP = rl / 64; // pointer for current long to edit
		final int lastP = (ru+1) / 64;
		final long finalOriginal = ret[lastP]; // original log at the ru location.

		// in this case the longs does not line up, and we therefore need to line them up.
		// LOG.error(longToBits(ret[retP]));
		// first mask out previous and then continue
		// mask by shifting two times (easier than constructing a mask)
		ret[retP] = (ret[retP] << invRemainder) >>> invRemainder;
		// LOG.error(longToBits(ret[retP]));
		// middle full 64 bit overwrite no need to mask first.
		// do not include last (it has to be specially handled)
		// LOG.error("Forloop");
		for(int j = 0; j < ov.length - 1; j++) {
			final long v = ov[j];
			// LOG.error(longToBits(v));
			// LOG.error(longToBits((v << remainder)));
			ret[retP] = ret[retP] ^ (v << remainder);
			// LOG.error(longToBits(ret[retP]));
			retP++;
			// LOG.error(longToBits(ret[retP]));
			ret[retP] = v >>> invRemainder;
			// LOG.error(longToBits(ret[retP]));
		}
		// LOG.error("ForLoop end");

		final long v = ov[ov.length - 1];
		final long last = v << remainder;
		final long re = ret[retP];
		// LOG.error(remainderEnd);
		// LOG.error(remainder);
		// LOG.error(invRemainder);
		// LOG.error(longToBits(prevLastExtracted));
		// LOG.error(longToBits(v));
		// LOG.error(longToBits(last));
		// LOG.error(longToBits(re));

		ret[retP] = last ^ re;

		// LOG.error(longToBits(ret[retP]));
		// ret[retP] = ret[retP] ^ (v << remainder);
		retP++;
		// LOG.error[]
		if(retP < ret.length && retP <= lastP) { // aka there is a remainder
			// LOG.error("Tail tail");

			// final long previousLast = ret[retP];
			// LOG.error(longToBits(previousLast));
			ret[retP] = v >>> invRemainder;
			// LOG.error(longToBits(ret[retP]));
			// ret[retP] = ret[retP] ^ (previousLast >>> remainderEnd) << remainderEnd;
			// LOG.error(longToBits(ret[retP]));
		}
		
		ret[lastP] = (ret[lastP] << remainderEndInv) >>> remainderEndInv;
		ret[lastP] = ret[lastP] ^ (finalOriginal >>> remainderEnd) << remainderEnd;
		
		return ret;
	}

	@Override
	public void setNz(int rl, int ru, Array<Boolean> value) {
		if(value instanceof BitSetArray) {
			throw new NotImplementedException();
		}
		else {

			boolean[] data2 = ((BooleanArray) value)._data;
			for(int i = rl; i < ru + 1; i++)
				if(data2[i])
					_data.set(i, data2[i]);
		}
	}

	@Override
	public void append(String value) {
		append(Boolean.parseBoolean(value));
	}

	@Override
	public void append(Boolean value) {
		_data.set(_size, value);
		_size++;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.BITSET.ordinal());
		long[] internals = _data.toLongArray();
		out.writeInt(internals.length);
		for(int i = 0; i < internals.length; i++)
			out.writeLong(internals[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		long[] internalLong = new long[in.readInt()];
		for(int i = 0; i < internalLong.length; i++)
			internalLong[i] = in.readLong();
		_data = BitSet.valueOf(internalLong);
	}

	@Override
	public Array<Boolean> clone() {
		long[] d = _data.toLongArray();
		int ln = d.length;
		long[] nd = Arrays.copyOf(d, ln);
		BitSet nBS = BitSet.valueOf(nd);
		return new BitSetArray(nBS, _size);
	}

	@Override
	public Array<Boolean> slice(int rl, int ru) {
		return new BitSetArray(_data.get(rl, ru), ru - rl);
	}

	@Override
	public Array<Boolean> sliceTransform(int rl, int ru, ValueType vt) {
		return slice(rl, ru);
	}

	@Override
	public void reset(int size) {
		_data = new BitSet();
		_size = size;
	}

	@Override
	public byte[] getAsByteArray(int nRow) {
		// over allocating here.. we could maybe bit pack?
		ByteBuffer booleanBuffer = ByteBuffer.allocate(nRow);
		booleanBuffer.order(ByteOrder.nativeOrder());
		// TODO: fix inefficient transfer 8 x bigger.
		// We should do bit unpacking on the python side.
		for(int i = 0; i < nRow; i++)
			booleanBuffer.put((byte) (_data.get(i) ? 1 : 0));
		return booleanBuffer.array();
	}

	@Override
	public ValueType getValueType() {
		return ValueType.BOOLEAN;
	}

	@Override
	public ValueType analyzeValueType() {
		return ValueType.BOOLEAN;
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.BITSET;
	}

	@Override
	public long getInMemorySize() {
		long size = super.getInMemorySize() + 8; // object header + object reference
		size += MemoryEstimates.bitSetCost(_size);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		long size = 1 + 4;
		size += _data.toLongArray().length * 8;
		return size;
	}

	@Override
	protected Array<?> changeTypeBitSet() {
		return clone();
	}

	@Override
	protected Array<?> changeTypeBoolean() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++)
			// if ever relevant use next set bit instead.
			// to increase speed, but it should not be the case in general
			ret[i] = _data.get(i);

		return new BooleanArray(ret);
	}

	@Override
	protected Array<?> changeTypeDouble() {
		double[] ret = new double[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data.get(i) ? 1.0 : 0.0;
		return new DoubleArray(ret);
	}

	@Override
	protected Array<?> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data.get(i) ? 1.0f : 0.0f;
		return new FloatArray(ret);
	}

	@Override
	protected Array<?> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data.get(i) ? 1 : 0;
		return new IntegerArray(ret);
	}

	@Override
	protected Array<?> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = _data.get(i) ? 1L : 0L;
		return new LongArray(ret);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size - 1; i++)
			sb.append((_data.get(i) ? 1 : 0) + ",");
		sb.append(_data.get(_size - 1) ? 1 : 0);
		sb.append("]");
		return sb.toString();
	}

	public static String longToBits(long l) {
		String bits = Long.toBinaryString(l);
		StringBuilder sb = new StringBuilder(64);
		for(int i = 0; i < 64 - bits.length(); i++) {
			sb.append('0');
		}
		sb.append(bits);
		return sb.toString();
	}
}
