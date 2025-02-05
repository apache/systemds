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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;

public class MapToZero extends AMapToData {

	private static final long serialVersionUID = -8065234231282619923L;

	private final int _size;

	public MapToZero(int size) {
		super(1);
		_size = size;
	}

	@Override
	public MAP_TYPE getType() {
		return MapToFactory.MAP_TYPE.ZERO;
	}

	@Override
	public int getIndex(int n) {
		return 0;
	}

	@Override
	public void fill(int v) {
		// do nothing
	}

	@Override
	public long getInMemorySize() {
		return getInMemorySize(0);
	}

	public static long getInMemorySize(int dataLength) {
		return 16 + 4;
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4;
	}

	@Override
	public void set(int n, int v) {
		// do nothing
	}

	@Override
	public void set(int l, int u, int off, AMapToData tm){
		// do nothing
	}

	@Override
	public int setAndGet(int n, int v) {
		return 0;
	}

	@Override
	public int size() {
		return _size;
	}

	@Override
	public void replace(int v, int r) {
		// do nothing
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(MAP_TYPE.ZERO.ordinal());
		out.writeInt(_size);
	}

	protected static MapToZero readFields(DataInput in) throws IOException {
		return new MapToZero(in.readInt());
	}

	@Override
	public int getUpperBoundValue() {
		return 0;
	}

	@Override
	public int[] getCounts(int[] ret) {
		ret[0] = size();
		return ret;
	}

	@Override
	public void preAggregateDDC_DDCSingleCol(AMapToData tm, double[] td, double[] v) {
		final int sz = size();
		for(int r = 0; r < sz; r++)
			v[0] += td[tm.getIndex(r)];

	}

	@Override
	public void preAggregateDDC_DDCMultiCol(AMapToData tm, IDictionary td, double[] v, int nCol) {
		final int sz = size();
		for(int r = 0; r < sz; r++)
			td.addToEntry(v, tm.getIndex(r), 0, nCol);
	}

	@Override
	public void copyInt(int[] d, int start, int end) {
		// do nothing
	}

	@Override
	public void copyBit(MapToBit d) {
		// do nothing
	}

	@Override
	public AMapToData resize(int unique) {
		// do nothing
		return this;
	}

	@Override
	public int countRuns() {
		return 1;
	}

	@Override
	public AMapToData slice(int l, int u) {
		return new MapToZero(u - l);
	}

	@Override
	public AMapToData append(AMapToData t) {
		if(t instanceof MapToZero)
			return new MapToZero(_size + t.size());
		else
			throw new NotImplementedException("Not implemented append on Bit map different type");
	}

	@Override
	public AMapToData appendN(IMapToDataGroup[] d) {
		int p = 0; // pointer
		boolean allZ = true;
		for(IMapToDataGroup gd : d) {
			AMapToData m = gd.getMapToData();

			p += m.size();
			if(!(m instanceof MapToZero))
				allZ = false;
		}

		if(!allZ)
			throw new RuntimeException("Not supported combining different types of map");

		return new MapToZero(p);
	}

	@Override
	public boolean equals(AMapToData e) {
		return e instanceof MapToZero && //
			_size == ((MapToZero) e)._size;
	}
}
