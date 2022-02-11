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
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToByte extends AMapToData {

	private static final long serialVersionUID = -2498505439667351828L;

	protected final byte[] _data;

	public MapToByte(int unique, int size) {
		super(Math.min(unique, 256));
		_data = new byte[size];
	}

	protected MapToByte(int unique, byte[] data) {
		super(unique);
		_data = data;
	}

	protected MapToUByte toUByte() {
		return new MapToUByte(getUnique(), _data);
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.BYTE;
	}

	@Override
	public int getIndex(int n) {
		return _data[n] & 0xFF;
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (byte) v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	public static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.byteArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = (byte) v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.BYTE.ordinal());
		writeBytes(out);
	}

	protected void writeBytes(DataOutput out) throws IOException {
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeByte(_data[i]);
	}

	protected static MapToByte readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final byte[] data = new byte[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readByte();
		return new MapToByte(unique, data);
	}

	@Override
	public void replace(int v, int r) {
		byte cv = (byte) v;
		byte rv = (byte) r;
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
	}

	@Override
	public void copy(AMapToData d) {
		if(d instanceof MapToChar) {
			char[] dd = ((MapToChar) d).getChars();
			for(int i = 0; i < size(); i++)
				_data[i] = (byte) dd[i];
		}
		else
			super.copy(d);
	}

	protected void preAggregateDenseToRowBy8(double[] mV, int off, double[] preAV, int cl, int cu) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[_data[rc] & 0xFF] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8) {
			final int id1 = _data[rc] & 0xFF, id2 = _data[rc + 1] & 0xFF, id3 = _data[rc + 2] & 0xFF, id4 = _data[rc + 3] & 0xFF,
				id5 = _data[rc + 4] & 0xFF, id6 = _data[rc + 5] & 0xFF, id7 = _data[rc + 6] & 0xFF,
				id8 = _data[rc + 7] & 0xFF;
			preAV[id1] += mV[off];
			preAV[id2] += mV[off + 1];
			preAV[id3] += mV[off + 2];
			preAV[id4] += mV[off + 3];
			preAV[id5] += mV[off + 4];
			preAV[id6] += mV[off + 5];
			preAV[id7] += mV[off + 6];
			preAV[id8] += mV[off + 7];
		}
	}

	@Override
	protected void preAggregateDenseSingleRow(double[] mV, int off, double[] preAV, int cl, int cu) {
		if(cu - cl > 64)
			// Have tried with 4 and 16, but 8 is empirically best
			preAggregateDenseToRowBy8(mV, off, preAV, cl, cu);
		else
			super.preAggregateDenseSingleRow(mV, off, preAV, cl, cu);
	}

	// @Override
	// protected void preAggregateDenseMultiRow(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu) {
	// final int nVal = getUnique();
	// final DenseBlock db = m.getDenseBlock();
	// if(db.isContiguous()) {
	// final double[] mV = m.getDenseBlockValues();
	// final int nCol = m.getNumColumns();
	// for(int c = cl; c < cu; c++) {
	// final int idx = getIndex(c);
	// final int start = c + nCol * rl;
	// final int end = c + nCol * ru;
	// for(int offOut = idx, off = start; off < end; offOut += nVal, off += nCol) {
	// preAV[offOut] += mV[off];
	// }
	// }
	// }
	// else
	// throw new NotImplementedException();
	// }

	@Override
	public final void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, AOffset indexes) {
		indexes.preAggregateDenseMap(m, preAV, rl, ru, cl, cu, getUnique(), _data);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes) {
		indexes.preAggregateSparseMap(sb, preAV, rl, ru, getUnique(), _data);
	}

	@Override
	public int getUpperBoundValue() {
		return 255;
	}

	// @Override
	// public int[] getCounts(int[] counts) {
	// final int sz = size();
	// for(int i = 0; i < sz; i++)
	// counts[getIndex(i)]++;
	// return counts;
	// }

	// @Override
	// protected void preAggregateDDC_DDCSingleCol(AMapToData tm, double[] td, double[] v) {
	// for(int r = 0; r < _data.length; r++)
	// v[_data[r] & 0xFF] += td[tm.getIndex(r)];
	// }

	// @Override
	// protected void preAggregateDDC_DDCMultiCol(AMapToData tm, ADictionary td, double[] v, int nCol) {
	// for(int r = 0; r < _data.length; r++)
	// td.addToEntry(v, tm.getIndex(r), _data[r] & 0xFF, nCol);
	// }

	// @Override
	// protected void preAggregateSDCZ_SDCZSingleCol(AMapToData tm, double[] td, double[] dv, AOffsetIterator itThat,
	// AOffsetIterator itThis, int tSize, int size) {

	// int i = 0, j = 0;

	// // main preAggregate process
	// while(i < tSize && j < size) {
	// final int tv = itThat.value();
	// final int v = itThis.value();
	// if(tv == v) {
	// final int fr = tm.getIndex(i);
	// dv[_data[j] & 0xFF] += td[fr];
	// itThat.next();
	// itThis.next();
	// i++;
	// j++;
	// }
	// else if(tv < v) {
	// itThat.next();
	// i++;
	// }
	// else {
	// itThis.next();
	// j++;
	// }
	// }

	// // Remaining part (very small so not really main performance bottleneck)
	// preAggregateSDCZ_SDCZMultiCol_tail(tm, this, new Dictionary(td), dv, 1, itThat, itThis, tSize, size, i, j);

	// }
}
