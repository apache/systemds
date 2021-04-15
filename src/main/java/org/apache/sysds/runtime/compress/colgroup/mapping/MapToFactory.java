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

public class MapToFactory {
	public static IMapToData create(int size, int numTuples) {
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

	public static IMapToData readIn(int size, DataInput in, int numTuples) throws IOException {
		if(numTuples <= 1)
			return new MapToBit(size).readFields(in);
		else if(numTuples <= 255)
			return new MapToByte(size).readFields(in);
		else if(numTuples <= Character.MAX_VALUE)
			return new MapToChar(size).readFields(in);
		else
			return new MapToInt(size).readFields(in);
	}

	public static long getExactSizeOnDisk(int size, int numTuples) {
		if(numTuples <= 1)
			return 4 + size / 8 + (size % 8 > 1 ? 1 : 0);
		else if(numTuples <= 256)
			return size;
		else if(numTuples <= Character.MAX_VALUE)
			return size * 2;
		else
			return size * 4;
	}
}
