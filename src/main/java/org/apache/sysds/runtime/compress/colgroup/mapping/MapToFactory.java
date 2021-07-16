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
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public final class MapToFactory {

	protected static final Log LOG = LogFactory.getLog(MapToFactory.class.getName());

	public enum MAP_TYPE {
		BIT, BYTE, CHAR, INT;
	}

	public static AMapToData create(ABitmap map) {
		return create(map.getNumRows(), map);
	}

	public static AMapToData create(int size, ABitmap map) {
		if(map == null || map.isEmpty())
			return null;

		boolean zeros = map.getNumOffsets() < size;
		return create(size, zeros, map.getOffsetList());
	}

	public static AMapToData create(int size, boolean zeros, IntArrayList[] values) {
		AMapToData _data = MapToFactory.create(size, values.length + (zeros ? 1 : 0));
		if(zeros)
			_data.fill(values.length);

		for(int i = 0; i < values.length; i++) {
			IntArrayList tmpList = values[i];
			final int sz = tmpList.size();
			for(int k = 0; k < sz; k++)
				_data.set(tmpList.get(k), i);
		}
		return _data;
	}

	public static AMapToData create(int size, int numTuples) {
		if(numTuples <= 1)
			return new MapToBit(numTuples, size);
		else if(numTuples < 256)
			return new MapToByte(numTuples, size);
		else if(numTuples < Character.MAX_VALUE)
			return new MapToChar(numTuples, size);
		else
			return new MapToInt(numTuples, size);
	}

	public static long estimateInMemorySize(int size, int numTuples) {
		if(numTuples <= 1)
			return MapToBit.getInMemorySize(size);
		else if(numTuples < 256)
			return MapToByte.getInMemorySize(size);
		else if(numTuples < Character.MAX_VALUE)
			return MapToChar.getInMemorySize(size);
		else
			return MapToInt.getInMemorySize(size);
	}

	public static AMapToData readIn(DataInput in) throws IOException {
		MAP_TYPE t = MAP_TYPE.values()[in.readByte()];
		switch(t) {
			case BIT:
				return MapToBit.readFields(in);
			case BYTE:
				return MapToByte.readFields(in);
			case CHAR:
				return MapToChar.readFields(in);
			case INT:
				return MapToInt.readFields(in);
			default:
				throw new DMLCompressionException("Unknown Map type.");
		}
	}

	public static AMapToData join(AMapToData left, AMapToData right) {
		if(left == null)
			return right;
		else if(right == null)
			return left;
		final int nVL = left.getUnique();
		final int nVR = right.getUnique();
		final int size = left.size();
		final int maxUnique = nVL * nVR;
		if(size != right.size())
			throw new DMLCompressionException("Invalid input maps to join, must contain same number of rows");

		try {
			return computeJoin(left, right, size, nVL, maxUnique);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Joining failed max unique expected:" + maxUnique, e);
		}
	}

	private static AMapToData computeJoin(AMapToData left, AMapToData right, int size, int nVL, int maxUnique) {
		AMapToData tmp = create(size, maxUnique);
		return computeJoinUsingLinearizedMap(tmp, left, right, size, nVL, maxUnique);
	}

	private static AMapToData computeJoinUsingLinearizedMap(AMapToData tmp, AMapToData left, AMapToData right, int size,
		int nVL, int maxUnique) {
		int[] map = new int[maxUnique];
		int newUID = 1;
		for(int i = 0; i < size; i++) {
			final int nv = left.getIndex(i) + right.getIndex(i) * nVL;
			final int mapV = map[nv];
			if(mapV == 0) {
				tmp.set(i, newUID - 1);
				map[nv] = newUID++;
			}
			else 
				tmp.set(i, mapV - 1);
		}

		tmp.setUnique(newUID-1);
		// LOG.error(Arrays.toString(map));
		return tmp;
	}
}
