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
import java.util.BitSet;

import org.apache.sysds.utils.MemoryEstimates;

public class MapToBit implements IMapToData {

	private BitSet _data;

	public MapToBit(int size){
		_data = new BitSet(size);
	}

	@Override
	public int getIndex(int n) {
		return _data.get(n)? 1: 0;
	}

	@Override
	public void fill(int v) {
		if(v == 1)
			_data.flip(0, _data.length());
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.size());
	}

	public static long getInMemorySize(int dataLength){
		long size = 16; // object header
		size += MemoryEstimates.bitSetCost(dataLength);
		return size;
	}

	@Override
	public void set(int n, int v) {
		_data.set(n, v == 1);
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		long[] internals =  _data.toLongArray();
		out.writeInt(internals.length);
		for(int i = 0; i < internals.length; i++)
			out.writeLong(internals[i]);
	}

	@Override
	public MapToBit readFields(DataInput in) throws IOException {
		long[] internalLong = new long[in.readInt()];
		for(int i = 0; i < internalLong.length; i++)
			internalLong[i] = in.readLong();
		
		_data = BitSet.valueOf(internalLong);
		return this;
	}
}
