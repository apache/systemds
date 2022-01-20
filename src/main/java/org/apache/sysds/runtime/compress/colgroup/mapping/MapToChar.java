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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.MemoryEstimates;

public class MapToChar extends AMapToData {

	private static final long serialVersionUID = 6315708056775476541L;

	private final char[] _data;

	public MapToChar(int unique, int size) {
		super(Math.min(unique, Character.MAX_VALUE + 1));
		_data = new char[size];
	}

	public MapToChar(int unique, char[] data) {
		super(unique);
		_data = data;
	}

	@Override
	public int getIndex(int n) {
		return _data[n];
	}

	@Override
	public void fill(int v) {
		Arrays.fill(_data, (char) v);
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(_data.length);
	}

	protected static long getInMemorySize(int dataLength) {
		long size = 16 + 8; // object header + object reference
		size += MemoryEstimates.charArrayCost(dataLength);
		return size;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 4 + _data.length * 2;
	}

	@Override
	public void set(int n, int v) {
		_data[n] = (char) v;
	}

	@Override
	public int size() {
		return _data.length;
	}

	@Override
	public void replace(int v, int r) {
		char cv = (char) v;
		char rv = (char) r;
		for(int i = 0; i < size(); i++)
			if(_data[i] == cv)
				_data[i] = rv;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.CHAR.ordinal());
		out.writeInt(getUnique());
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeChar(_data[i]);
	}

	protected static MapToChar readFields(DataInput in) throws IOException {
		int unique = in.readInt();
		final int length = in.readInt();
		final char[] data = new char[length];
		for(int i = 0; i < length; i++)
			data[i] = in.readChar();
		return new MapToChar(unique, data);
	}

	protected char[] getChars() {
		return _data;
	}

	private void preAggregateDenseToRowBy8(double[] mV, double[] preAV, int cl, int cu, int off) {
		final int h = (cu - cl) % 8;
		off += cl;
		for(int rc = cl; rc < cl + h; rc++, off++)
			preAV[_data[rc]] += mV[off];
		for(int rc = cl + h; rc < cu; rc += 8, off += 8) {
			int id1 = _data[rc], id2 = _data[rc + 1], id3 = _data[rc + 2], id4 = _data[rc + 3], id5 = _data[rc + 4],
				id6 = _data[rc + 5], id7 = _data[rc + 6], id8 = _data[rc + 7];
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
	protected void preAggregateDenseToRow(double[] mV, int off, double[] preAV, int cl, int cu) {
		if(cu - cl > 1000)
			preAggregateDenseToRowBy8(mV, preAV, cl, cu, off);
		else {
			off += cl;
			for(int rc = cl; rc < cu; rc++, off++)
				preAV[_data[rc]] += mV[off];
		}
	}

	@Override
	protected void preAggregateDenseRows(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu) {
		final int nVal = getUnique();
		final DenseBlock db = m.getDenseBlock();
		if(db.isContiguous()) {
			final double[] mV = m.getDenseBlockValues();
			final int nCol = m.getNumColumns();
			for(int c = cl; c < cu; c++) {
				final int idx = getIndex(c);
				final int start = c + nCol * rl;
				final int end = c + nCol * ru;
				for(int offOut = idx, off = start; off < end; offOut += nVal, off += nCol) {
					preAV[offOut] += mV[off];
				}
			}
		}
		else
			throw new NotImplementedException();
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAV, int rl, int ru, int cl, int cu, AOffset indexes) {
		indexes.preAggregateDenseMap(m, preAV, rl, ru, cl, cu, getUnique(), _data);
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAV, int rl, int ru, AOffset indexes) {
		indexes.preAggregateSparseMap(sb, preAV, rl, ru, getUnique(), _data);
	}

	@Override
	public int getUpperBoundValue() {
		return Character.MAX_VALUE;
	}

}
