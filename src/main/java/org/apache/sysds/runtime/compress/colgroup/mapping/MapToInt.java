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

package org.apache.sysds.runtime.compress.colgroup.mapping;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.sysds.utils.MemoryEstimates;

public class MapToInt implements IMapToData {

	private int[] _data;

	public MapToInt(int size) {
		_data = new int[size];
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16; // object header
		size += MemoryEstimates.intArrayCost(dataLength);
		return size;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = v;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		for(int i = 0; i < _data.length; i++)
			out.writeInt(_data[i]);
	}

	@Override
	public MapToInt readFields(DataInput in) throws IOException {
		for(int i = 0; i < _data.length; i++)
			_data[i] = in.readInt();
		return this;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nDataLength: " + this._data.length);
		sb.append(Arrays.toString(this._data));
		return sb.toString();
	}

}
