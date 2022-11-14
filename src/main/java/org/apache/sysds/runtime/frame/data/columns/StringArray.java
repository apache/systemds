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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;

public class StringArray extends Array<String> {
	private String[] _data = null;

	public StringArray(String[] data) {
		_data = data;
		_size = _data.length;
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
	}

	@Override
	public void set(int rl, int ru, Array<String> value) {
		set(rl, ru, value, 0);
	}

	@Override
	public void set(int rl, int ru, Array<String> value, int rlSrc) {
		System.arraycopy(((StringArray) value)._data, rlSrc, _data, rl, ru - rl + 1);
	}

	@Override
	public void setNz(int rl, int ru, Array<String> value) {
		String[] data2 = ((StringArray) value)._data;
		for(int i = rl; i < ru + 1; i++)
			if(data2[i] != null)
				_data[i] = data2[i];
	}

	@Override
	public void append(String value) {
		if(_data.length <= _size)
			_data = Arrays.copyOf(_data, newSize());
		_data[_size++] = value;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		for(int i = 0; i < _size; i++)
			out.writeUTF((_data[i] != null) ? _data[i] : "");
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_size = _data.length;
		for(int i = 0; i < _size; i++) {
			String tmp = in.readUTF();
			_data[i] = (!tmp.isEmpty()) ? tmp : null;
		}
	}

	@Override
	public Array<String> clone() {
		return new StringArray(Arrays.copyOf(_data, _size));
	}

	@Override
	public Array<String> slice(int rl, int ru) {
		return new StringArray(Arrays.copyOfRange(_data, rl, ru + 1));
	}

	@Override
	public void reset(int size) {
		if(_data.length < size)
			_data = new String[size];
		_size = size;
	}

	@Override
	public byte[] getAsByteArray(int nRow) {
		throw new NotImplementedException("Not Implemented getAsByte for string");
	}

	@Override
	public ValueType getValueType() {
		return ValueType.STRING;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder(_data.length * 5 + 2);
		sb.append(super.toString() + ":[");
		for(int i = 0; i < _size-1; i++)
			sb.append(_data[i]  + ",");
		sb.append(_data[_size-1]);
		sb.append("]");
		return sb.toString();
	}
}
