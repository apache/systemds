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

package org.apache.sysds.test.component.compress.mapping;

import static org.junit.Assert.fail;

import java.util.Random;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToCharPByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public class MappingTestUtil {
	@SuppressWarnings("fallthrough")
	protected static AMapToData[] getAllHigherVersions(AMapToData m) {
		AMapToData[] ret = new AMapToData[getTypeSize(m.getType())];
		int idx = 0;
		switch(m.getType()) {
			case ZERO:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.ZERO);
			case BIT:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.UBYTE);
			case UBYTE:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.BYTE);
			case BYTE:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.CHAR);
			case CHAR:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.CHAR_BYTE);
			case CHAR_BYTE:
				ret[idx++] = MapToFactory.resizeForce(m, MAP_TYPE.INT);
			case INT:
				// none
		}
		return ret;
	}

	protected static int getTypeSize(MAP_TYPE t) {
		switch(t) {
			case INT:
				return 0;
			case CHAR_BYTE:
				return 1;
			case CHAR:
				return 2;
			case BYTE:
				return 3;
			case UBYTE:
				return 4;
			case BIT:
				return 5;
			case ZERO:
				return 6;
			default:
				fail("Unknown type: " + t);
				return -1;
		}
	}

	protected static AMapToData createRandomMap(int len, int nUnique, Random r) {
		AMapToData m = MapToFactory.create(len, nUnique);
		for(int i = 0; i < len; i++)
			m.set(i, r.nextInt(nUnique));
		return m;
	}

	protected static AOffset createRandomOffset(int offRange, int nRows, Random r) {
		IntArrayList offs = new IntArrayList();
		int off = r.nextInt(offRange);
		if(off < nRows)
			offs.appendValue(off);
		while(off < nRows) {
			off += r.nextInt(offRange) + 1;
			if(off < nRows)
				offs.appendValue(off);
		}
		return OffsetFactory.createOffset(offs);
	}


	public static int getUpperBoundValue(MAP_TYPE t) {
		switch(t) {
			case ZERO:
				return 0;
			case BIT:
				return 1;
			case UBYTE:
				return 127;
			case BYTE:
				return 255;
			case CHAR:
				return Character.MAX_VALUE;
			case CHAR_BYTE:
				return MapToCharPByte.max - 1;
			case INT:
				return Integer.MAX_VALUE;
			default:
				throw new DMLCompressionException("Unsupported type " + t);
		}
	}
}
