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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.util.CommonThreadPool;

/** Interface for the factory design pattern for construction all AMapToData. */
public interface MapToFactory {
	static final Log LOG = LogFactory.getLog(MapToFactory.class.getName());

	/** The different supported types of mappings. */
	public enum MAP_TYPE {
		ZERO, BIT, UBYTE, BYTE, CHAR, CHAR_BYTE, INT;
	}

	/**
	 * Construct a mapping with the given values contained. The size is the length of the int array given.
	 * 
	 * @param values  The values contained.
	 * @param nUnique The number of unique expected to be contained (is not verified.)
	 * @return An appropriate AMapToData
	 */
	public static AMapToData create(int[] values, int nUnique) {
		AMapToData _data = create(values.length, nUnique);
		_data.copyInt(values);
		return _data;
	}

	/**
	 * Construct a mapping with the given values contained.
	 * 
	 * Only copies the values from the array given until size.
	 * 
	 * @param size    The number of elements to take from the values array.
	 * @param values  The values contained.
	 * @param nUnique The number of unique expected to be contained (is not verified.)
	 * @return An appropriate AMapToData
	 */
	public static AMapToData create(int size, int[] values, int nUnique) {
		AMapToData _data = create(size, nUnique);
		_data.copyInt(values);
		return _data;
	}

	public static AMapToData create(int unique, IntArrayList values) {
		AMapToData _data = create(values.size(), unique);
		_data.copyInt(values.extractValues());
		return _data;
	}

	public static AMapToData create(int size, int[] values, int nUnique, int k) {
		AMapToData _data = create(size, nUnique);
		ExecutorService pool = CommonThreadPool.get(k);
		int blk = Math.max((values.length / k), 1024);
		blk -= blk % 64; // ensure long size
		List<Future<?>> tasks = new ArrayList<>();
		for(int i = 0; i < values.length; i += blk){
			int start = i;
			int end = Math.min(i + blk, values.length);
			tasks.add(pool.submit(() -> _data.copyInt(values, start, end)));
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
	public static AMapToData create(final int size, final int numTuples) {
		if(numTuples <= 1)
			return new MapToZero(size);
		else if(numTuples == 2 && size > 32)
			return new MapToBit(numTuples, size);
		else if(numTuples <= 127)
			return new MapToUByte(numTuples, size);
		else if(numTuples <= 256)
			return new MapToByte(numTuples, size);
		else if(numTuples <= Character.MAX_VALUE + 1)
			return new MapToChar(numTuples, size);
		else if(numTuples <= MapToCharPByte.max)
			return new MapToCharPByte(numTuples, size);
		else
			return new MapToInt(numTuples, size);
	}

	/**
	 * Allocate a specific type of map. Note that once in use it is recommended to set the number of unique values.
	 * 
	 * @param size The size to allocate
	 * @param t    The mapping type.
	 * @return An AMapToData allocation
	 */
	public static AMapToData create(final int size, final MAP_TYPE t) {
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
			default:
				return new MapToInt(size);
		}
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
			default:
				ret = new MapToInt(numTuples, size);
		}
		ret.copy(d);
		return ret;
	}

	/**
	 * Estimate the size in memory of a MapToFactory.
	 * 
	 * @param size      The size of the mapping
	 * @param numTuples The number of unique values to be supported by the mapping
	 * @return The size in number of bytes.
	 */
	public static long estimateInMemorySize(int size, int numTuples) {
		if(numTuples <= 1)
			return MapToZero.getInMemorySize(size);
		else if(numTuples == 2 && size > 32)
			return MapToBit.getInMemorySize(size);
		else if(numTuples <= 256)
			return MapToByte.getInMemorySize(size);
		else if(numTuples <= Character.MAX_VALUE + 1)
			return MapToChar.getInMemorySize(size);
		else if(numTuples <= MapToCharPByte.max)
			return MapToCharPByte.getInMemorySize(size);
		else
			return MapToInt.getInMemorySize(size);
	}

	/**
	 * General interface to read in an AMapToData.
	 * 
	 * @param in The data input to read from
	 * @return The parsed AMapToData
	 * @throws IOException If there is complications or errors in reading.
	 */
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
			default:
				return MapToInt.readFields(in);
		}
	}
}
