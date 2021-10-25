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

import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToInt extends AMapToData {

	private static final long serialVersionUID = -5557070920888782274L;

	private final int[] _data;

	public MapToInt(int unique, int size) {
		super(unique);
		_data = new int[size];
	}

	private MapToInt(int unique, int[] data) {
		super(unique);
		_data = data;
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
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.intArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length * 4;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void replace(int v, int r) {
		for(int i = 0; i < size(); i++)
			if(_data[i] == v)
				_data[i] = r;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.INT.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeInt(_data[i]);
	}

	public static MapToInt readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final int[] data = new int[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readInt();
		return new MapToInt(unique, data);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, MatrixBlock pre, int rl, int ru, int cl, int cu) {
		final int nRow = m.getNumColumns();
		final int nVal = pre.getNumColumns();
		final double[] preAV = pre.getDenseBlockValues();
		final double[] mV = m.getDenseBlockValues();
		final int blockSize = 4000;
		for(int block = cl; block < cu; block += blockSize) {
			final int blockEnd = Math.min(block + blockSize, nRow);
			for(int rowLeft = rl, offOut = 0; rowLeft < ru; rowLeft++, offOut += nVal) {
				final int offLeft = rowLeft * nRow;
				for(int rc = block; rc < blockEnd; rc++) {
					final int idx = _data[rc];
					preAV[offOut + idx] += mV[offLeft + rc];
				}
			}
		}
	}

	@Override
	public int getUpperBoundValue() {
		return Integer.MAX_VALUE;
	}
}
