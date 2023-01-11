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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class BitSetArray extends Array<Boolean> {

	private static final boolean useVectorizedKernel = true;

	/** Vectorized "words" containing all the bits set */
	long[] _data;

	protected BitSetArray(int size) {
		super(size);
		_data = new long[size / 64 + 1];
	}

	public BitSetArray(boolean[] data) {
		super(data.length);
		if(data.length == 11)
			throw new DMLRuntimeException("Invalid length");
		_data = new long[_size / 64 + 1];
		// set bits.
		for(int i = 0; i < data.length; i++)
			if(data[i]) // slightly more efficient to check.
				set(i, true);
	}

	public BitSetArray(long[] data, int size) {
		super(size);
		_data = data;
		if(_size > _data.length * 64)
			throw new DMLRuntimeException("Invalid allocation long array must be long enough");
		if(_data.length > _size / 64 + 1)
			throw new DMLRuntimeException(
				"Invalid allocation long array must not be to long" + _data.length + " " + _size + " " + (size / 64 + 1));
	}

	public BitSetArray(BitSet data, int size) {
		super(size);
		_data = toLongArrayPadded(data, size);
	}

	public BitSet get() {
		return BitSet.valueOf(_data);
	}

	public long[] getLongs() {
		return _data;
	}

	@Override
	public Boolean get(int index) {
		int wIdx = index >> 6; // same as divide by 64 bit faster
		return (_data[wIdx] & (1L << index)) != 0;
	}

	@Override
	public void set(int index, Boolean value) {
		set(index, value != null && value);
	}

	public void set(int index, boolean value) {
		int wIdx = index >> 6; // same as divide by 64 bit faster
		if(value)
			_data[wIdx] |= (1L << index);
		else
			_data[wIdx] &= ~(1L << index);
	}

	@Override
	public void set(int index, double value) {
		set(index, value == 1.0);
	}

	@Override
	public void set(int index, String value) {
		set(index, BooleanArray.parseBoolean(value));
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++)
			set(i, UtilFunctions.objectToBoolean(vt, value.get(i)));

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
				set(i, value.get(off));
	}

	private void setVectorized(int rl, int ru, BitSetArray value, int rlSrc) {
		final int rangeLength = ru - rl + 1;
		final BitSetArray v = value.slice(rlSrc, rangeLength + rlSrc);
		final long[] otherValues = v.getLongs();
		setVectorizedLongs(rl, ru, otherValues);
	}

	private void setVectorizedLongs(int rl, int ru, long[] ov) {
		final long remainder = rl % 64L;
		if(remainder == 0)
			setVectorizedLongsNoOffset(rl, ru, ov);
		else
			setVectorizedLongsWithOffset(rl, ru, ov);
	}

	private void setVectorizedLongsNoOffset(int rl, int ru, long[] ov) {
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;
		final int last = ov.length - 1;
		int retP = rl / 64;

		// assign all full.
		for(int j = 0; j < last; j++, retP++)
			_data[retP] = ov[j];

		// handle tail.
		if(remainderEnd != 0) {
			// clear ret in the area.
			final long r = (_data[retP] >>> remainderEnd) << remainderEnd;
			final long v = (ov[last] << remainderEndInv) >>> remainderEndInv;
			// assign ret in the area.
			_data[retP] = r ^ v;
		}
		else
			_data[retP] = ov[last];
	}

	private void setVectorizedLongsWithOffset(int rl, int ru, long[] ov) {
		final long remainder = rl % 64L;
		final long invRemainder = 64L - remainder;
		final int last = ov.length - 1;
		final int lastP = (ru + 1) / 64;
		final long finalOriginal = _data[lastP]; // original log at the ru location.

		int retP = rl / 64; // pointer for current long to edit

		// first mask out previous and then continue
		// mask by shifting two times (easier than constructing a mask)
		_data[retP] = (_data[retP] << invRemainder) >>> invRemainder;

		// middle full 64 bit overwrite no need to mask first.
		// do not include last (it has to be specially handled)
		for(int j = 0; j < last; j++) {
			final long v = ov[j];
			_data[retP] = _data[retP] ^ (v << remainder);
			retP++;
			_data[retP] = v >>> invRemainder;
		}

		_data[retP] = (ov[last] << remainder) ^ _data[retP];
		retP++;
		if(retP < _data.length && retP <= lastP) // aka there is a remainder
			_data[retP] = ov[last] >>> invRemainder;

		// reassign everything outside range of ru.
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;
		_data[lastP] = (_data[lastP] << remainderEndInv) >>> remainderEndInv;
		_data[lastP] = _data[lastP] ^ (finalOriginal >>> remainderEnd) << remainderEnd;

	}

	@Override
	public void setNz(int rl, int ru, Array<Boolean> value) {
		if(value instanceof BooleanArray) {
			final boolean[] data2 = ((BooleanArray) value)._data;
			for(int i = rl; i <= ru; i++)
				if(data2[i])
					set(i, data2[i]);
		}
		else {
			// TODO add an vectorized setNz.
			for(int i = rl; i <= ru; i++) {
				final boolean v = value.get(i);
				if(v)
					set(i, v);
			}
		}
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		final ValueType vt = value.getValueType();
		for(int i = rl; i <= ru; i++) {
			boolean v = UtilFunctions.objectToBoolean(vt, value.get(i));
			if(v)
				set(i, v);
		}
	}

	@Override
	public void append(String value) {
		append(BooleanArray.parseBoolean(value));
	}

	@Override
	public BitSetArray append(Array<Boolean> other) {
		final int endSize = this._size + other.size();
		final BitSetArray retBS = new BitSetArray(endSize);
		retBS.set(0, this._size - 1, this, 0);
		retBS.set(this._size, endSize - 1, other, 0);
		return retBS;
	}

	@Override
	public void append(Boolean value) {
		if(_data.length * 64 < _size + 1)
			_data = Arrays.copyOf(_data, newSize());
		set(_size, value);
		_size++;
	}

	@Override
	public int newSize() {
		return _data.length * 2;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.BITSET.ordinal());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeLong(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_data = new long[in.readInt()];
		for(int i = 0; i < _data.length; i++)
			_data[i] = in.readLong();
	}

	@Override
	public BitSetArray clone() {
		return new BitSetArray(Arrays.copyOf(_data, _data.length), _size);
	}

	@Override
	public BitSetArray slice(int rl, int ru) {
		return ru - rl > 30 // if over threshold
			? sliceVectorized(rl, ru) // slice vectorized
			: sliceSimple(rl, ru); // slice via get
	}

	private BitSetArray sliceSimple(int rl, int ru) {
		final boolean[] ret = new boolean[ru - rl];
		for(int i = rl, off = 0; i < ru; i++, off++)
			ret[off] = get(i);
		return new BitSetArray(ret);
	}

	private BitSetArray sliceVectorized(int rl, int ru) {

		final long[] ret = new long[(ru - rl) / 64 + 1];

		final long BitIndexMask = (1 << 6) - 1L;
		final long lastMask = 0xffffffffffffffffL >>> -ru;

		// targetWords
		final int tW = ((ru - rl - 1) >>> 6) + 1;
		// sourceIndex
		int sI = rl >> 6;

		boolean aligned = (rl & BitIndexMask) == 0;

		// all but last
		if(aligned) {
			for(int i = 0; i < tW - 1; i++, sI++) {
				ret[i] = _data[sI];
			}
		}
		else {
			for(int i = 0; i < tW - 1; i++, sI++) {
				ret[i] = (_data[sI] >>> rl) | (_data[sI + 1] << -rl);
			}
		}

		// last
		ret[tW - 1] = //
			(((ru - 1) & BitIndexMask)) < (rl & BitIndexMask) //
				? (_data[sI] >>> rl) | (_data[sI + 1] & lastMask) << -rl //
				: (_data[sI] & lastMask) >>> rl;

		return new BitSetArray(ret, ru - rl);
	}

	@Override
	public void reset(int size) {
		_data = new long[size / 64 + 1];
		_size = size;
	}

	@Override
	public byte[] getAsByteArray() {
		// over allocating here.. we could maybe bit pack?
		ByteBuffer booleanBuffer = ByteBuffer.allocate(_size);
		booleanBuffer.order(ByteOrder.nativeOrder());
		// TODO: fix inefficient transfer 8 x bigger.
		// We should do bit unpacking on the python side.
		for(int i = 0; i < _size; i++)
			booleanBuffer.put((byte) (get(i) ? 1 : 0));
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
		return estimateInMemorySize(_size);
	}

	public static long estimateInMemorySize(int nRow) {
		long size = baseMemoryCost(); // object header + object reference
		size += MemoryEstimates.longArrayCost(nRow >> 6 + 1);
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		long size = 1 + 4;
		size += _data.length * 8;
		return size;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		return this;
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		boolean[] ret = new boolean[size()];
		for(int i = 0; i < size(); i++)
			// if ever relevant use next set bit instead.
			// to increase speed, but it should not be the case in general
			ret[i] = get(i);

		return new BooleanArray(ret);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		double[] ret = new double[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i) ? 1.0 : 0.0;
		return new DoubleArray(ret);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		float[] ret = new float[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i) ? 1.0f : 0.0f;
		return new FloatArray(ret);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		int[] ret = new int[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i) ? 1 : 0;
		return new IntegerArray(ret);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		long[] ret = new long[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i) ? 1L : 0L;
		return new LongArray(ret);
	}

	@Override
	protected Array<String> changeTypeString() {
		String[] ret = new String[size()];
		for(int i = 0; i < size(); i++)
			ret[i] = get(i).toString();
		return new StringArray(ret);
	}

	@Override
	public void fill(String value) {
		fill(BooleanArray.parseBoolean(value));
	}

	@Override
	public void fill(Boolean value) {
		for(int i = 0; i < _size / 64 + 1; i++)
			_data[i] = value ? -1L : 0L;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size; i++)
			sb.append((get(i) ? 1 : 0));
		sb.append("]");
		return sb.toString();
	}

	@Override
	public double getAsDouble(int i) {
		return get(i) ? 1.0 : 0.0;
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
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
