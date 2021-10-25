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

import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToBit extends AMapToData {

	private static final long serialVersionUID = -8065234231282619923L;

	private final BitSet _data;
	private final int _size;

	public MapToBit(int unique, int size) {
		super(unique);
		_data = new BitSet(size);
		_size = size;
	}

	private MapToBit(int unique, BitSet d, int size) {
		super(unique);
		_data = d;
		_size = size;
	}

	@Override
	public int getIndex(int n) {
		return _data.get(n) ? 1 : 0;
	}

	@Override
	public void fill(int v) {
		_data.set(0, _size, true);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.size());
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8 + 4; // object header + object reference + int size
		size += MemoryEstimates.bitSetCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		final int dSize = _data.size();
		long size = 1 + 4 + 4 + 4; // base variables
		size += (dSize / 64) * 8; // all longs except last
		// size += (dSize % 64 == 0 ? 0 : 8); // last long
		return size;
	}

	@Override
	public void set(int n, int v) {
		_data.set(n, v == 1);
	}

	@Override
	public int size() {
		return _size;
	}

	@Override
	public void replace(int v, int r) {
		// Note that this method assume that replace is called correctly.
		if(v == 0) // set all to 1
			_data.set(0, size(), true);
		else // set all to 0
			_data.clear();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		long[] internals = _data.toLongArray();
		out.writeByte(MAP_TYPE.BIT.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_size);
		out.writeInt(internals.length);
		for(int i = 0; i < internals.length; i++)
			out.writeLong(internals[i]);
	}

	public static MapToBit readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		int size = in.readInt();
		long[] internalLong = new long[in.readInt()];
		for(int i = 0; i < internalLong.length; i++)
			internalLong[i] = in.readLong();

		return new MapToBit(unique, BitSet.valueOf(internalLong), size);
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
				for(int rc = block; rc < blockEnd; rc++)
					preAV[_data.get(rc) ? offOut + 1 : offOut] += mV[offLeft + rc];
			}
		}
	}

	@Override
	public int getUpperBoundValue() {
		return 1;
	}
}
