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
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public interface MapToFactory {
	static final Log LOG = LogFactory.getLog(MapToFactory.class.getName());

	public enum MAP_TYPE {
		ZERO, BIT, UBYTE, BYTE, CHAR, CHAR_BYTE, INT;
	}

	public static AMapToData create(int size, ABitmap ubm) {
		if(ubm == null)
			return null;
		return create(size, ubm.containsZero(), ubm.getOffsetList());
	}

	public static AMapToData create(int size, boolean zeros, IntArrayList[] values) {
		AMapToData _data = create(size, values.length + (zeros ? 1 : 0));

		if(zeros)
			_data.fill(values.length);

		for(int i = 0; i < values.length; i++) {
			final IntArrayList tmpList = values[i];
			final int sz = tmpList.size();
			for(int k = 0; k < sz; k++){
				_data.set(tmpList.get(k), i);
			}
		}
		return _data;
	}

	public static AMapToData create(int size, int[] values, int nUnique) {
		AMapToData _data = create(size, nUnique);
		_data.copyInt(values);
		return _data;
	}

	/**
	 * Create and allocate a map with the given size and support for upto the num tuples argument of values
	 * 
	 * @param size      The number of cells to allocate
	 * @param numTuples The maximum value to be able to represent inside the map.
	 * @return A new map
	 */
	public static AMapToData create(final int size, final int numTuples) {
		if(numTuples <= 1)
			return new MapToZero(size);
		else if(numTuples == 2 && size > 32)
			return new MapToBit(numTuples, size);
		else if(numTuples <= 127)
			return new MapToUByte(numTuples, size);
		else if(numTuples <= 256)
			return new MapToByte(numTuples, size);
		else if(numTuples <= ((int) Character.MAX_VALUE) + 1)
			return new MapToChar(numTuples, size);
		else if(numTuples <= MapToCharPByte.max)
			return new MapToCharPByte(numTuples, size);
		else
			return new MapToInt(numTuples, size);
	}

	public static AMapToData create(int size, MAP_TYPE t) {
		switch(t) {
			case ZERO:
				return new MapToZero(size);
			case BIT:
				return new MapToBit(size);
			case UBYTE:
				return new MapToUByte(size);
			case BYTE:
				return new MapToByte(size);
			case CHAR:
				return new MapToChar(size);
			case CHAR_BYTE:
				return new MapToCharPByte(size);
			case INT:
				return new MapToInt(size);
			default:
				throw new DMLCompressionException("Unsupported type " + t);
		}
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
		return d.resize(numTuples);
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
			case ZERO:
				return new MapToZero(size);
			case BIT:
				ret = new MapToBit(numTuples, size);
				break;
			case UBYTE:
				ret = new MapToUByte(numTuples, size);
				break;
			case BYTE:
				ret = new MapToByte(numTuples, size);
				break;
			case CHAR:
				ret = new MapToChar(numTuples, size);
				break;
			case CHAR_BYTE:
				ret = new MapToCharPByte(numTuples, size);
				break;
			case INT:
				ret = new MapToInt(numTuples, size);
				break;
			default:
				throw new DMLCompressionException("Unsupported type of map " + t);
		}
		ret.copy(d);
		return ret;
	}

	public static long estimateInMemorySize(int size, int numTuples) {
		if(numTuples <= 1)
			return MapToZero.getInMemorySize(size);
		else if(numTuples == 2 && size > 32)
			return MapToBit.getInMemorySize(size);
		else if(numTuples <= 256)
			return MapToByte.getInMemorySize(size);
		else if(numTuples <= ((int) Character.MAX_VALUE) + 1)
			return MapToChar.getInMemorySize(size);
		else if(numTuples <= MapToCharPByte.max)
			return MapToCharPByte.getInMemorySize(size);
		else
			return MapToInt.getInMemorySize(size);
	}

	public static AMapToData readIn(DataInput in) throws IOException {
		MAP_TYPE t = MAP_TYPE.values()[in.readByte()];
		switch(t) {
			case ZERO:
				return MapToZero.readFields(in);
			case BIT:
				return MapToBit.readFields(in);
			case UBYTE:
				return MapToUByte.readFields(in);
			case BYTE:
				return MapToByte.readFields(in);
			case CHAR:
				return MapToChar.readFields(in);
			case CHAR_BYTE:
				return MapToCharPByte.readFields(in);
			case INT:
				return MapToInt.readFields(in);
			default:
				throw new DMLCompressionException("unsupported type " + t);
		}
	}
}
