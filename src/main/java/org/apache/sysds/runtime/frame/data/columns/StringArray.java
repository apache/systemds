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
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.utils.MemoryEstimates;

public class StringArray extends Array<String> {
	private String[] _data;

	private long materializedSize = -1L;

	private StringArray(int nRow) {
		this(new String[nRow]);
	}

	public StringArray(String[] data) {
		super(data.length);
		_data = data;
	}

	private StringArray(String[] data, long materializedSize) {
		this(data);
		this.materializedSize = materializedSize;
	}

	public String[] get() {
		return _data;
	}

	@Override
	public String get(int index) {
		return _data[index];
	}

	@Override
	public void set(int index, String value) {
		_data[index] = value;
		materializedSize = -1;
	}

	@Override
	public void set(int index, double value) {
		_data[index] = Double.toString(value);
		materializedSize = -1;
	}

	@Override
	public void setFromOtherType(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			final Object v = value.get(i);
			if(v != null)
				_data[i] = v.toString();
			else
				_data[i] = null;
		}
		materializedSize = -1;
	}

	@Override
	public void set(int rl, int ru, Array<String> value, int rlSrc) {
		try {
			// try system array copy.
			// but if it does not work, default to get.
			System.arraycopy(value.get(), rlSrc, _data, rl, ru - rl + 1);
		}
		catch(Exception e) {
			super.set(rl, ru, value, rlSrc);
		}
		finally {
			materializedSize = -1;
		}
	}

	@Override
	public void setNz(int rl, int ru, Array<String> value) {
		String[] data2 = ((StringArray) value)._data;
		for(int i = rl; i <= ru; i++)
			if(data2[i] != null)
				_data[i] = data2[i];
		materializedSize = -1;
	}

	@Override
	public void setFromOtherTypeNz(int rl, int ru, Array<?> value) {
		for(int i = rl; i <= ru; i++) {
			Object v = value.get(i);
			if(v != null)
				_data[i] = v.toString();
		}
		materializedSize = -1;
	}

	@Override
	public void append(String value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = value;
		materializedSize = -1;
	}

	@Override
	public Array<String> append(Array<String> other) {
		final int endSize = this._size + other.size();
		final String[] ret = new String[endSize];
		System.arraycopy(_data, 0, ret, 0, this._size);
		System.arraycopy(other.get(), 0, ret, this._size, other.size());
		return new StringArray(ret);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.STRING.ordinal());
		out.writeLong(getInMemorySize());

		for(int i = 0; i < _size; i++)
			out.writeUTF((_data[i] != null) ? _data[i] : "");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		materializedSize = in.readLong();
		for(int i = 0; i < _size; i++) {
			String tmp = in.readUTF();
			_data[i] = tmp.isEmpty() ? null : tmp;
		}
	}

	protected static StringArray read(DataInput in, int nRow) throws IOException {
		final StringArray arr = new StringArray(nRow);
		arr.readFields(in);
		return arr;
	}

	@Override
	public Array<String> clone() {
		return new StringArray(Arrays.copyOf(_data, _size), materializedSize);
	}

	@Override
	public Array<String> slice(int rl, int ru) {
		return new StringArray(Arrays.copyOfRange(_data, rl, ru));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size || _data.length > 2 * size)
			_data = new String[size];
		else
			for(int i = 0; i < size; i++)
				_data[i] = null;
		_size = size;
		materializedSize = -1;
	}

	@Override
	public byte[] getAsByteArray() {
		throw new NotImplementedException("Not Implemented getAsByte for string");
	}

	/**
	 * Python interface to extract strings from systemds.
	 * 
	 * @param r the index to extract
	 * @return The value in bytes for py4j
	 */
	public byte[] getIndexAsBytes(int r) {
		if(_data[r] != null)
			return _data[r].getBytes();
		else
			return null;
	}

	@Override
	public ValueType getValueType() {
		return ValueType.STRING;
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		ValueType state = ValueType.UNKNOWN;
		boolean nulls = false;
		for(int i = 0; i < Math.min(maxCells, _size); i++) {
			final ValueType c = FrameUtil.isType(_data[i], state);
			if(c == ValueType.STRING)
				return new Pair<>(ValueType.STRING, false);
			else if(c == ValueType.UNKNOWN)
				nulls = true;
			else
				state = ValueType.getHighestCommonTypeSafe(state, c);
		}
		return new Pair<>(state, nulls);
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.STRING;
	}

	@Override
	public BitSetArray getNulls() {
		BitSetArray n = new BitSetArray(_size);
		for(int i = 0; i < _size; i++)
			if(_data[i] != null)
				n.set(i, true);
		return n;
	}

	@Override
	public long getInMemorySize() {
		if(materializedSize != -1)
			return materializedSize;

		long size = super.getInMemorySize(); // object header + object reference
		size += MemoryEstimates.stringArrayCost(_data);
		size += 8; // estimated size cache
		return materializedSize = size;
	}

	@Override
	public long getExactSerializedSize() {
		long si = 1 + 8; // byte identifier and long size
		for(String s : _data)
			si += IOUtilFunctions.getUTFSize(s);
		return si;
	}

	@Override
	protected Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int rl, int ru) {
		String firstNN = _data[rl];
		int i = rl + 1;
		while(firstNN == null && i < ru) {
			firstNN = _data[i++];
		}

		if(firstNN == null)
			return ret;// all values were null. therefore we have an easy time retuning an empty boolean array.
		else if(firstNN.toLowerCase().equals("true") || firstNN.toLowerCase().equals("false"))
			return changeTypeBooleanStandardBitSet(ret, rl, ru);
		else if(firstNN.equals("0") || firstNN.equals("1") || firstNN.equals("1.0") || firstNN.equals("0.0"))
			return changeTypeBooleanNumericBitSet(ret, rl, ru);
		else if(firstNN.toLowerCase().equals("t") || firstNN.toLowerCase().equals("f"))
			return changeTypeBooleanCharacterBitSet(ret, rl, ru);
		else
			throw new DMLRuntimeException("Not supported type of Strings to change to Booleans value: " + firstNN);
	}

	@Override
	protected Array<Boolean> changeTypeBoolean(Array<Boolean> ret, int rl, int ru) {
		String firstNN = _data[rl];
		int i = rl + 1;
		while(firstNN == null && i < ru) {
			firstNN = _data[i++];
		}

		if(firstNN == null)
			return ret;// all values were null. therefore we have an easy time retuning an empty boolean array.
		else if(firstNN.toLowerCase().equals("true") || firstNN.toLowerCase().equals("false"))
			return changeTypeBooleanStandardArray(ret, rl, ru);
		else if(firstNN.equals("0") || firstNN.equals("1") || firstNN.equals("1.0") || firstNN.equals("0.0"))
			return changeTypeBooleanNumericArray(ret, rl, ru);
		else if(firstNN.toLowerCase().equals("t") || firstNN.toLowerCase().equals("f"))
			return changeTypeBooleanCharacterArray(ret, rl, ru);
		else
			throw new DMLRuntimeException("Not supported type of Strings to change to Booleans value: " + firstNN);
	}

	protected Array<Boolean> changeTypeBooleanStandardBitSet(Array<Boolean> ret, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null)
				ret.set(i, Boolean.parseBoolean(s));
		}
		return ret;
	}

	protected Array<Boolean> changeTypeBooleanStandardArray(Array<Boolean> retA, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null)
				retA.set(i, Boolean.parseBoolean(s));
		}
		return retA;
	}

	protected Array<Boolean> changeTypeBooleanCharacterBitSet(Array<Boolean> ret, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null) {
				if(isTrueCharacter(s.charAt(0)))
					ret.set(i, true);
				else if(isFalseCharacter(s.charAt(0)))
					ret.set(i, false);
				else
					throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);

			}
		}
		return ret;
	}

	protected Array<Boolean> changeTypeBooleanCharacterArray(Array<Boolean> retA, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null) {

				if(isTrueCharacter(s.charAt(0)))
					retA.set(i, true);
				else if(isFalseCharacter(s.charAt(0)))
					retA.set(i, false);
				else
					throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);
			}

		}
		return retA;
	}

	private boolean isTrueCharacter(char a) {
		return a == 'T' || a == 't';
	}

	private boolean isFalseCharacter(char a) {
		return a == 'F' || a == 'f';
	}

	protected Array<Boolean> changeTypeBooleanNumericBitSet(Array<Boolean> ret, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null) {
				if(s.length() > 1) {
					final boolean zero = s.equals("0.0");
					final boolean one = s.equals("1.0");
					if(zero | one)
						ret.set(i, one);
					else
						throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);

				}
				else {
					final boolean zero = s.charAt(0) == '0';
					final boolean one = s.charAt(0) == '1';
					if(zero | one)
						ret.set(i, one);
					else
						throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);
				}
			}
		}
		return ret;
	}

	protected Array<Boolean> changeTypeBooleanNumericArray(Array<Boolean> retA, int rl, int ru) {

		for(int i = rl; i < ru; i++) {
			final String s = _data[i];
			if(s != null) {
				if(s.length() > 1) {
					final boolean zero = s.equals("0.0");
					final boolean one = s.equals("1.0");
					if(zero | one)
						retA.set(i, one);
					else
						throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);

				}
				else {
					final boolean zero = s.charAt(0) == '0';
					final boolean one = s.charAt(0) == '1';
					if(zero | one)
						retA.set(i, one);
					else
						throw new DMLRuntimeException("Unable to change to Boolean from String array, value: " + s);
				}
			}

		}
		return retA;
	}

	@Override
	protected Array<Double> changeTypeDouble(Array<Double> retA, int l, int u) {
		try {
			for(int i = l; i < u; i++)
				retA.set(i, DoubleArray.parseDouble(_data[i]));
			return retA;
		}
		catch(Exception e) {
			Pair<ValueType, Boolean> t = analyzeValueType();
			if(t.getKey() == ValueType.BOOLEAN)
				changeType(ValueType.BOOLEAN).changeType(retA, l, u);
			else
				throw e;
			return retA;
		}
	}

	@Override
	protected Array<Float> changeTypeFloat(Array<Float> retA, int l, int u) {
		for(int i = l; i < u; i++)
			retA.set(i, FloatArray.parseFloat(_data[i]));
		return retA;
	}

	@Override
	protected Array<Integer> changeTypeInteger(Array<Integer> retA, int l, int u) {
		String firstNN = _data[l];
		int i = l + 1;
		while(firstNN == null && i < u) {
			firstNN = _data[i++];
		}

		if(firstNN == null)
			return retA; // no non zero values.

		if(firstNN.contains("."))
			changeTypeIntegerFloatString(retA, l, u);
		else
			changeTypeIntegerNormal(retA, l, u);
		return retA;
	}

	protected void changeTypeIntegerFloatString(Array<Integer> ret, int l, int u) {
		for(int i = l; i < u; i++) {
			final String s = _data[i];
			if(s != null)
				ret.set(i, parseSignificantFloat(s));
		}
	}

	protected int parseSignificantFloat(String s) {
		final int len = s.length();
		int v = 0; // running sum of Significant
		if(len == 0)
			return v;
		int c = 0; // current character
		char ch = s.charAt(c);
		final boolean isNegative = ch == '-';
		if(isNegative || ch == '+') {
			c++;
		}
		do {
			ch = s.charAt(c++);
			final int cc = ch - '0';
			if(ch == '.')
				break;
			else if(cc < 10)
				v = 10 * v + cc;
			else
				throw new NumberFormatException(s);
		}
		while(c < len);

		for(; c < len; c++) {
			if(s.charAt(c) != '0')
				throw new NumberFormatException(s);
		}
		return isNegative ? -v : v;
	}

	protected void changeTypeIntegerNormal(Array<Integer> ret, int l, int u) {
		for(int i = l; i < u; i++) {
			final String s = _data[i];
			if(s != null)
				ret.set(i, parseInt(s));
		}
	}

	protected int parseInt(String s) {
		final int len = s.length();
		int v = 0; // running sum of Significant
		if(len == 0)
			return v;
		int c = 0;
		char ch = s.charAt(c);
		final boolean isNegative = ch == '-';
		if(isNegative || ch == '+') {
			c++;
		}
		do {
			ch = s.charAt(c++);
			final int cc = ch - '0';
			if(cc < 10)
				v = 10 * v + cc;
			else
				throw new NumberFormatException(s);
		}
		while(c < len);

		return isNegative ? -v : v;
	}

	@Override
	protected Array<Long> changeTypeLong(Array<Long> retA, int l, int u) {
		for(int i = l; i < u; i++) {
			final String s = _data[i];
			if(s != null)
				retA.set(i, Long.parseLong(s));
		}
		return retA;
	}

	@Override
	protected Array<Object> changeTypeHash64(Array<Object> retA, int l, int u) {
		for(int i = l; i < u; i++)
			retA.set(i, _data[i]);
		return retA;
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> retA, int l, int u) {
		for(int i = l; i < u; i++)
			retA.set(i, _data[i]);
		return retA;
	}

	@Override
	public Array<Character> changeTypeCharacter(Array<Character> retA, int l, int u) {
		for(int i = l; i < u; i++) {
			final String s = _data[i];
			if(s != null)
				retA.set(i, s.charAt(0));
		}
		return retA;
	}

	@Override
	public Array<String> changeTypeString(Array<String> retA, int l, int u) {
		String[] ret = (String[]) retA.get();
		for(int i = l; i < u; i++)
			ret[i] = _data[i];
		return retA;
	}

	@Override
	public Pair<Integer, Integer> getMinMaxLength() {
		int minLength = Integer.MAX_VALUE;
		int maxLength = 0;
		for(int i = 0; i < _size; i++) {
			if(_data[i] == null)
				continue;
			int l = _data[i].length();
			minLength = minLength < l ? minLength : l;
			maxLength = maxLength > l ? maxLength : l;

		}
		return new Pair<>(minLength, maxLength);
	}

	@Override
	public void fill(String value) {
		Arrays.fill(_data, value);
		materializedSize = -1;
	}

	@Override
	public double getAsDouble(int i) {
		if(_data[i] != null && !_data[i].isEmpty()) {
			return getAsDouble(_data[i]);
		}
		else {
			return 0.0;
		}
	}

	@Override
	public double getAsNaNDouble(int i) {
		if(_data[i] != null && !_data[i].isEmpty()) {
			return getAsDouble(_data[i]);
		}
		else {
			return Double.NaN;
		}
	}

	private static double getAsDouble(String s) {
		try {
			return DoubleArray.parseDouble(s);
		}
		catch(Exception e) {
			String ls = s.toLowerCase();
			if(ls.equals("true") || ls.equals("t"))
				return 1;
			else if(ls.equals("false") || ls.equals("f"))
				return 0;
			else
				throw e; // for efficiency
				// throw new DMLRuntimeException("Unable to change to double: " + s, e);
		}
	}

	@Override
	public boolean isShallowSerialize() {
		final long s = getInMemorySize();
		return _size < 100 || s / _size < 256;
	}

	@Override
	public boolean isEmpty() {
		for(int i = 0; i < _size; i++)
			if(_data[i] != null && !_data[i].equals("0"))
				return false;
		return true;
	}

	@Override
	public boolean containsNull() {
		for(int i = 0; i < _size; i++)
			if(_data[i] == null)
				return true;
		return false;
	}

	@Override
	public Array<String> select(int[] indices) {
		final String[] ret = new String[indices.length];
		for(int i = 0; i < indices.length; i++)
			ret[i] = _data[indices[i]];
		return new StringArray(ret);
	}

	@Override
	public Array<String> select(boolean[] select, int nTrue) {
		final String[] ret = new String[nTrue];
		int k = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				ret[k++] = _data[i];
		return new StringArray(ret);
	}

	@Override
	public final boolean isNotEmpty(int i) {
		return _data[i] != null && !_data[i].equals("0");
	}

	@Override
	protected HashMapToInt<String> createRecodeMap(int estimate, ExecutorService pool, int k) throws InterruptedException, ExecutionException {
		try {
			HashMapToInt<String> map = new HashMapToInt<String>((int) Math.min((long) estimate * 2, size()));
			for(int i = 0; i < size(); i++) {
				Object val = get(i);
				if(val != null) {
					String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(val.toString());
					map.put(tmp[0], Integer.parseInt(tmp[1]));
				}
			}
			return map;
		}
		catch(Exception e) {
			return super.createRecodeMap(estimate, pool, k);
		}
	}

	@Override
	public double hashDouble(int idx) {
		if(_data[idx] != null)
			return _data[idx].hashCode();
		else
			return Double.NaN;
	}

	@Override
	public boolean equals(Array<String> other) {
		if(other instanceof StringArray)
			return Arrays.equals(_data, ((StringArray) other)._data);
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_size * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size - 1; i++)
			sb.append(_data[i] + ",");
		sb.append(_data[_size - 1]);
		sb.append("]");
		return sb.toString();
	}
}
