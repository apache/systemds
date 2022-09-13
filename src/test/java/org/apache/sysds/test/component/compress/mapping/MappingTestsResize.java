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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingTestsResize {

	public final int seed;
	public final MAP_TYPE type;
	public final int size;

	private AMapToData m;
	private int[] expected;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(MAP_TYPE t : MAP_TYPE.values()) {
			tests.add(new Object[] {1, t, 13, false});
			tests.add(new Object[] {1, t, 632, false});
		}
		return tests;
	}

	public MappingTestsResize(int seed, MAP_TYPE type, int size, boolean fill) {
		this.seed = seed;
		this.type = type;
		this.size = size;
		try{

			final int max = Math.min(MappingTestUtil.getUpperBoundValue(type),size);
			final int maxSmaller = Math.min(getMaxSmaller(type), size);
			expected = new int[size];
			m = MappingTests.genMap(MapToFactory.create(size, max), expected, maxSmaller, fill, seed);
		}
		catch(Exception e){
			e.printStackTrace();
			fail("Failed creating mapping resize test");
		}
	}

	@Test
	public void resize() {
		MappingTests.compare(MapToFactory.resize(m, getMaxSmaller(type)), m);
	}

	private int getMaxSmaller(MAP_TYPE type) {
		switch(type) {
			case BIT:
			case UBYTE:
				return 1;
			case BYTE:
				return 127;
			case CHAR:
				return (int) Math.pow(2, 8) - 1;
			default:
				return (int) Character.MAX_VALUE;
		}
	}

}
