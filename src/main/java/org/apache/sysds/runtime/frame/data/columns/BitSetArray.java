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
import java.util.BitSet;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.utils.MemoryEstimates;

public class BitSetArray extends Array<Boolean> {
	private BitSet _data;

	protected BitSetArray(int size) {
		_size = size;
		_data = new BitSet();
	}

	public BitSetArray(boolean[] data) {
		_size = data.length;
		_data = new BitSet(data.length);
		// set bits.
		for(int i = 0; i < data.length; i++)
			if(data[i])
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
		if(value != null)
			_data.set(index, value);
		else
			_data.set(index, false);
	}

	@Override
	public void set(int index, double value) {
		if(value == 0)
			_data.set(index, false);
		else
			_data.set(index, true);
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		throw new NotImplementedException();
	}

	@Override
	public void set(int rl, int ru, Array<Boolean> value, int rlSrc) {
		if(value instanceof BitSetArray) {
			BitSet other = (BitSet)value.get();
			BitSet otherRange = other.get(rlSrc, ru - rl + 1 - rlSrc);
			long[] otherValues = otherRange.toLongArray();
			long[] ret = _data.toLongArray();

			final int remainder = rl % 64;
			int retP = rl / 64;
			// if(remainder == 0 && ) // lining up hurray
				// throw new NotImplementedException();
			// else{ // not lining up. Blame 1000 row blocks.
				// first mask out previous and then continue
				// mask by shifting two times (easier than constructing a mask)
				ret[retP] = (ret[retP] << remainder) >>> remainder;

				// middle full 64 bit overwrite no need to mask first.
				// do not include last (it has to be specially handled)
				for(int j = 0; j < otherValues.length - 1; j++){
					long v = otherValues[j];
					ret[retP] = ret[retP] ^ (v << remainder);
					retP++;
					ret[retP] = v >>>(64 - remainder);
				}
				// last mask out previous and remember
				long v = otherValues[otherValues.length -1];
				ret[retP] = ret[retP] ^ (v << remainder);
				retP++;
				if(retP < ret.length){ // aka there is a remainder
					long previousLast = ret[retP];
					ret[retP] = v >>> (64 - remainder);
					ret[retP] = ret[retP]  ^ previousLast << remainder;
				}
			// }
			_data = BitSet.valueOf(ret); // set data.
		}
		else {
			for(int i = rl, off = rlSrc; i <= ru; i++, off++)
				_data.set(i, value.get(off));
		}

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
		return new BitSetArray(BitSet.valueOf(_data.toLongArray()), _size);
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
	protected Array<?> changeTypeBoolean() {
		return clone();
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
}
