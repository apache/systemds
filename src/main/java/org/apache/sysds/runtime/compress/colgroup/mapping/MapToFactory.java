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

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MapToFactory {

	public enum MAP_TYPE {
		BIT, BYTE, CHAR, INT;
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
			return new MapToBit(size);
		else if(numTuples <= 256)
			return new MapToByte(size);
		else if(numTuples <= Character.MAX_VALUE)
			return new MapToChar(size);
		else
			return new MapToInt(size);
	}

	public static long estimateInMemorySize(int size, int numTuples) {
		if(numTuples <= 1)
			return MapToBit.getInMemorySize(size);
		else if(numTuples <= 256)
			return MapToByte.getInMemorySize(size);
		else if(numTuples <= Character.MAX_VALUE)
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
}
