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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MapToFactory {
	// protected static final Log LOG = LogFactory.getLog(MapToFactory.class.getName());

	public enum MAP_TYPE {
		BIT, BYTE, CHAR, INT;
	}

	public static AMapToData create(int size, ABitmap ubm) {
		if(ubm == null)
			return null;
		return create(size, ubm.containsZero(), ubm.getOffsetList());
	}

	public static AMapToData create(int size, boolean zeros, IntArrayList[] values) {
		AMapToData _data = MapToFactory.create(size, values.length + (zeros ? 1 : 0));

		if(zeros)
			_data.fill(values.length);

		for(int i = 0; i < values.length; i++) {
			final IntArrayList tmpList = values[i];
			final int sz = tmpList.size();
			for(int k = 0; k < sz; k++)
				_data.set(tmpList.get(k), i);
		}
		return _data;
	}

	/**
	 * Create and allocate a map with the given size and support for upto the num tuples argument of values
	 * 
	 * @param size      The number of cells to allocate
	 * @param numTuples The maximum value to be able to represent inside the map.
	 * @return A new map
	 */
	public static AMapToData create(int size, int numTuples) {
		if(numTuples <= 2)
			return new MapToBit(numTuples, size);
		else if(numTuples <= 256)
			return new MapToByte(numTuples, size);
		else if(numTuples <= ((int) Character.MAX_VALUE) + 1)
			return new MapToChar(numTuples, size);
		else
			return new MapToInt(numTuples, size);
	}

	/**
	 * Reshape the map, to a smaller instance if applicable.
	 * 
	 * Note that it returns the input if the input is the smallest representation that fits, otherwise it will return
	 * something that is smaller.
	 * 
	 * @param d         The Input mat to potentially reduce the size of.
	 * @param numTuples The number of tuples that should be in the resulting map
	 * @return The returned hopefully reduced map.
	 */
	public static AMapToData resize(AMapToData d, int numTuples) {
		final int size = d.size();
		AMapToData ret;
		if(d instanceof MapToBit)
			return d;
		else if(numTuples <= 2)
			ret = new MapToBit(numTuples, size);
		else if(d instanceof MapToByte)
			return d;
		else if(numTuples <= 256)
			ret = new MapToByte(numTuples, size);
		else if(d instanceof MapToChar)
			return d;
		else if(numTuples <= (int) Character.MAX_VALUE + 1)
			ret = new MapToChar(numTuples, size);
		else // then the input was int and reshapes to int
			return d;

		ret.copy(d);
		return ret;
	}

	/**
	 * Force the mapping into an other mapping type. This method is unsafe since if there is overflows in the
	 * conversions, they are not handled. Also if the change is into the same type a new map is allocated anyway.
	 * 
	 * @param d The map to resize.
	 * @param t The type to resize to.
	 * @return A new allocated mapToData with the specified type.
	 */
	public static AMapToData resizeForce(AMapToData d, MAP_TYPE t) {
		final int size = d.size();
		final int numTuples = d.getUnique();
		AMapToData ret;
		switch(t) {
			case BIT:
				ret = new MapToBit(numTuples, size);
				break;
			case BYTE:
				ret = new MapToByte(numTuples, size);
				break;
			case CHAR:
				ret = new MapToChar(numTuples, size);
				break;
			case INT:
			default:
				ret = new MapToInt(numTuples, size);
				break;
		}
		ret.copy(d);
		return ret;
	}

	public static long estimateInMemorySize(int size, int numTuples) {
		if(numTuples <= 2)
			return MapToBit.getInMemorySize(size);
		else if(numTuples <= 256)
			return MapToByte.getInMemorySize(size);
		else if(numTuples <= ((int) Character.MAX_VALUE) + 1)
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
			default:
				return MapToInt.readFields(in);
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
		final long maxUnique = (long) nVL * nVR;
		if(maxUnique > (long) Integer.MAX_VALUE)
			throw new DMLCompressionException(
				"Joining impossible using linearized join, since each side has a large number of unique values");
		if(size != right.size())
			throw new DMLCompressionException("Invalid input maps to join, must contain same number of rows");

		return computeJoin(left, right, size, nVL, (int) maxUnique);
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

		tmp.setUnique(newUID - 1);
		return tmp;
	}

	public static int getUpperBoundValue(MAP_TYPE t) {
		switch(t) {
			case BIT:
				return 1;
			case BYTE:
				return 255;
			case CHAR:
				return Character.MAX_VALUE;
			case INT:
			default:
				return Integer.MAX_VALUE;
		}
	}
}
