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
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

public class BitSetArray extends ABooleanArray {

	private static final boolean useVectorizedKernel = true;

	/** Vectorized "words" containing all the bits set */
	protected long[] _data;

	// private volatile int allTrue = -1;

	protected BitSetArray(int size) {
		this(new long[longSize(size)], size);
	}

	public BitSetArray(boolean[] data) {
		super(data.length);
		_data = new long[longSize(_size)];
		for(int i = 0; i < data.length; i++)
			if(data[i]) // slightly more efficient to check.
				set(i, true);
	}

	public BitSetArray(long[] data, int size) {
		super(size);
		_data = data;
		if(_size > _data.length * 64)
			throw new DMLRuntimeException("Invalid allocation long array must be long enough");
		if(_data.length > longSize(_size))
			throw new DMLRuntimeException("Invalid allocation long array must not be to long: " + _data.length + " "
				+ _size + " " + (longSize(_size)));
	}

	private static int longSize(int size) {
		return Math.max(size >> 6, 0) + 1;
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
		if(index >= _size)
			throw new ArrayIndexOutOfBoundsException(index);
		int wIdx = index >> 6; // same as divide by 64 bit faster
		return (_data[wIdx] & (1L << index)) != 0;
	}

	@Override
	public void set(int index, Boolean value) {
		set(index, value != null && value);
	}

	public synchronized void set(int index, boolean value) {
		int wIdx = index >> 6; // same as divide by 64 bit faster
		if(value)
			_data[wIdx] |= (1L << index);
		else
			_data[wIdx] &= ~(1L << index);
	}

	@Override
	public void setNullsFromString(int rl, int ru, Array<String> value) {

		final int rl64 = Math.min((rl / 64 + 1) * 64, ru);
		final int ru64 = (ru / 64) * 64;

		for(int i = rl; i < rl64; i++)
			set(i, value.get(i) != null);
		for(int i = rl64; i < ru64; i++)
			unsafeSet(i, value.get(i) != null);
		for(int i = ru64; i < ru; i++)
			set(i, value.get(i) != null);
	}

	private void unsafeSet(int index, boolean value) {
		int wIdx = index >> 6; // same as divide by 64 bit faster
		if(value)
			_data[wIdx] |= (1L << index);
		else
			_data[wIdx] &= ~(1L << index);
	}

	@Override
	public void set(int index, double value) {
		set(index, Math.round(value) == 1.0);
	}

	@Override
	public void set(int index, String value) {
		set(index, BooleanArray.parseBoolean(value));
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
		if(useVectorizedKernel && value instanceof BitSetArray && (ru - rl >= 64)) {
			try {
				// Try system array copy.
				// but if it does not work, default to get.
				setVectorized(rl, ru, (BitSetArray) value, rlSrc);
				return;
			}
			catch(Exception e) {
				// fall back to default
			}
		}
		// default
		super.set(rl, ru, value, rlSrc);
	}

	private void setVectorized(int rl, int ru, BitSetArray value, int rlSrc) {
		final int rangeLength = ru - rl + 1;
		final BitSetArray v = value.slice(rlSrc, rangeLength + rlSrc);
		final long[] otherValues = v.getLongs();
		setVectorizedLongs(rl, ru, otherValues);
	}

	private void setVectorizedLongs(int rl, int ru, long[] ov) {
		setVectorizedLongs(rl, ru, _data, ov);
	}

	public static void setVectorizedLongs(int rl, int ru, long[] ret, long[] ov) {

		final long remainder = rl % 64L;
		if(remainder == 0)
			setVectorizedLongsNoOffset(rl, ru, ret, ov);
		else
			setVectorizedLongsWithOffset(rl, ru, ret, ov);
	}

	private static void setVectorizedLongsNoOffset(int rl, int ru, long[] ret, long[] ov) {
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;
		final int last = ov.length - 1;
		int retP = rl / 64;

		// assign all full.
		for(int j = 0; j < last; j++, retP++)
			ret[retP] = ov[j];

		// handle tail.
		if(remainderEnd != 0) {
			// clear ret in the area.
			final long r = (ret[retP] >>> remainderEnd) << remainderEnd;
			final long v = (ov[last] << remainderEndInv) >>> remainderEndInv;
			// assign ret in the area.
			ret[retP] = r ^ v;
		}
		else
			ret[retP] = ov[last];
	}

	private static void setVectorizedLongsWithOffset(int rl, int ru, long[] ret, long[] ov) {
		final long remainder = rl % 64L;
		final long invRemainder = 64L - remainder;
		final int last = ov.length - 1;
		final int lastP = (ru + 1) / 64;
		final long finalOriginal = ret[lastP]; // original log at the ru location.

		int retP = rl / 64; // pointer for current long to edit

		// first mask out previous and then continue
		// mask by shifting two times (easier than constructing a mask)
		ret[retP] = (ret[retP] << invRemainder) >>> invRemainder;

		// middle full 64 bit overwrite no need to mask first.
		// do not include last (it has to be specially handled)
		for(int j = 0; j < last; j++) {
			final long v = ov[j];
			ret[retP] = ret[retP] ^ (v << remainder);
			retP++;
			ret[retP] = v >>> invRemainder;
		}

		ret[retP] = (ov[last] << remainder) ^ ret[retP];
		retP++;
		if(retP < ret.length && retP <= lastP) // aka there is a remainder
			ret[retP] = ov[last] >>> invRemainder;

		// reassign everything outside range of ru.
		final long remainderEnd = (ru + 1) % 64L;
		final long remainderEndInv = 64L - remainderEnd;
		ret[lastP] = (ret[lastP] << remainderEndInv) >>> remainderEndInv;
		ret[lastP] = ret[lastP] ^ (finalOriginal >>> remainderEnd) << remainderEnd;

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
	public Array<Boolean> append(Array<Boolean> other) {
		final int endSize = this._size + other.size();
		final Array<Boolean> retBS = ArrayFactory.allocateBoolean(endSize);
		retBS.set(0, this._size - 1, this);
		if(other instanceof OptionalArray) {
			retBS.set(this._size, endSize - 1, ((OptionalArray<Boolean>) other)._a);
			return OptionalArray.appendOther((OptionalArray<Boolean>) other, retBS);
		}
		else {
			retBS.set(this._size, endSize - 1, other);
			return retBS;
		}
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
		out.writeInt(_size);
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeLong(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = in.readInt();
		_data = new long[in.readInt()];
		for(int i = 0; i < _data.length; i++)
			_data[i] = in.readLong();
	}

	protected static BitSetArray read(DataInput in, int nRow) throws IOException {
		final BitSetArray arr = new BitSetArray(nRow);
		arr.readFields(in);
		return arr;
	}

	@Override
	public BitSetArray clone() {
		return new BitSetArray(Arrays.copyOf(_data, _size / 64 + 1), _size);
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

		return new BitSetArray(sliceVectorized(_data, rl, ru), ru - rl);
	}

	public static long[] sliceVectorized(long[] _data, int rl, int ru) {

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

		return ret;
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
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return new Pair<>(ValueType.BOOLEAN, false);
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
		long size = 1 + 4 + 4;
		size += _data.length * 8;
		return size;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u) {
		for(int i = l; i < u; i++)
			ret.set(i, get(i));
		return ret;
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> retA, int l, int u) {
		boolean[] ret = (boolean[]) retA.get();
		for(int i = l; i < u; i++)
			// if ever relevant use next set bit instead.
			// to increase speed, but it should not be the case in general
			ret[i] = get(i);
		return retA;
	}

	@Override
	protected Array<Double> changeTypeDouble(Array<Double> retA, int l, int u) {
		double[] ret = (double[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1.0 : 0.0;
		return retA;

	}

	@Override
	protected Array<Float> changeTypeFloat(Array<Float> retA, int l, int u) {
		float[] ret = (float[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1.0f : 0.0f;
		return retA;

	}

	@Override
	protected Array<Integer> changeTypeInteger(Array<Integer> retA, int l, int u) {
		int[] ret = (int[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1 : 0;
		return retA;

	}

	@Override
	protected Array<Long> changeTypeLong(Array<Long> retA, int l, int u) {
		long[] ret = (long[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1L : 0L;
		return new LongArray(ret);
	}

	@Override
	protected Array<Object> changeTypeHash64(Array<Object> retA, int l, int u) {
		long[] ret = ((HashLongArray) retA).getLongs();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1L : 0L;
		return retA;
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> retA, int l, int u) {
		int[] ret = ((HashIntegerArray) retA).getInts();
		for(int i = l; i < u; i++)
			ret[i] = get(i) ? 1 : 0;
		return retA;
	}

	@Override
	protected Array<String> changeTypeString(Array<String> retA, int l, int u) {
		String[] ret = (String[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = get(i).toString();
		return retA;

	}

	@Override
	public Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		char[] ret = (char[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = (char) (get(i) ? 1 : 0);
		return retA;

	}

	@Override
	public void fill(String value) {
		fill(BooleanArray.parseBoolean(value));
	}

	@Override
	public void fill(Boolean value) {
		value = value != null ? value : false;
		Arrays.fill(_data, value ? -1L : 0L);
	}

	@Override
	public double getAsDouble(int i) {
		return get(i) ? 1.0 : 0.0;
	}

	@Override
	public boolean isShallowSerialize() {
		return true;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _data.length; i++)
			if(_data[i] != 0L)
				return false;
		return true;
	}

	@Override
	public boolean isAllTrue() {
		for(int i = 0; i < _data.length; i++)
			if(_data[i] != -1L)
				return false;
		return true;
	}

	@Override
	public ABooleanArray select(int[] indices) {
		// TODO vectorize
		final boolean[] ret = new boolean[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = get(indices[i]);
		return new BitSetArray(ret);
	}

	@Override
	public ABooleanArray select(boolean[] select, int nTrue) {
		final boolean[] ret = new boolean[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = get(i);
		return new BitSetArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return get(i);
	}

	@Override
	public void findEmptyInverse(boolean[] select) {
		for(int i = 0; i < select.length; i++)
			if(!get(i))
				select[i] = true;
	}

	public static String longToBits(long l) {
		String bits = Long.toBinaryString(l);
		StringBuilder sb = new StringBuilder(64);
		for(int i = 0; i < 64 - bits.length(); i++)
			sb.append('0');
		sb.append(bits);
		return sb.toString();
	}

	@Override
	public double hashDouble(int idx) {
		return get(idx) ? 1.0 : 0.0;
	}

	@Override
	public ArrayCompressionStatistics statistics(int nSamples) {
		// Unlikely to compress so lets just say... no
		return new ArrayCompressionStatistics(1, //
			2, true, ValueType.BOOLEAN, false, FrameArrayType.DDC, getInMemorySize(), getInMemorySize() * 2, true);

	}

	@Override
	public boolean equals(Array<Boolean> other) {
		if(other instanceof BitSetArray)
			return Arrays.equals(_data, ((BitSetArray) other)._data);
		else
			return false;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size + 10);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size; i++)
			sb.append((get(i) ? 1 : 0));

		sb.append("]");
		return sb.toString();
	}
}
