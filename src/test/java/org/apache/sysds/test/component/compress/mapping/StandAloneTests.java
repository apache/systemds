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

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class StandAloneTests {

	protected static final Log LOG = LogFactory.getLog(StandAloneTests.class.getName());

	@Test
	public void testJoin_01() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {2, 4, 6, 8})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {3, 2, 0, 2, 0, 3, 1, 3, 1, 3});
		compare(c, new int[] {0, 1, 2, 1, 2, 0, 3, 0, 3, 0});
	}

	@Test
	public void testJoin_02() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, a);
		// compare(c, new int[] {1, 0, 0, 0, 0, 1, 1, 1, 1, 1});
		compare(c, new int[] {0, 1, 1, 1, 1, 0, 0, 0, 0, 0});
	}

	@Test
	public void testJoin_03() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {2, 0, 0, 0, 1, 2, 2, 2, 2, 2});
		compare(c, new int[] {0, 1, 1, 1, 2, 0, 0, 0, 0, 0});
	}

	@Test
	public void testJoin_04() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {2, 0, 0, 0, 1, 2, 2, 2, 2, 2});
		compare(c, new int[] {0, 1, 1, 1, 2, 0, 0, 0, 0, 0});
	}

	@Test
	public void testJoin_05() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3}), gen(new int[] {4})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {2, 0, 0, 0, 1, 2, 2, 2, 2, 2});
		compare(c, new int[] {0, 1, 1, 1, 2, 0, 0, 0, 0, 0});
	}

	@Test
	public void testJoin_06() {
		AMapToData a = MapToFactory.create(10, true,
			new IntArrayList[] {gen(new int[] {1, 2, 3}), gen(new int[] {4, 5})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {3, 0, 0, 0, 1, 2, 3, 3, 3, 3});
		compare(c, new int[] {0, 1, 1, 1, 2, 3, 0, 0, 0, 0});
	}

	@Test
	public void testJoin_07() {
		AMapToData a = MapToFactory.create(10, true,
			new IntArrayList[] {gen(new int[] {1, 2, 3}), gen(new int[] {4, 5}), gen(new int[] {6, 7})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, b);
		// compare(c, new int[] {4, 0, 0, 0, 1, 2, 3, 3, 4, 4});
		compare(c, new int[] {0, 1, 1, 1, 2, 3, 4, 4, 0, 0});
	}

	@Test(expected = RuntimeException.class)
	public void testInvalidArgument() {
		AMapToData a = MapToFactory.create(11, true,
			new IntArrayList[] {gen(new int[] {1, 2, 3}), gen(new int[] {4, 5}), gen(new int[] {6, 7})});
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		MapToFactory.join(a, b);
	}

	@Test(expected = DMLCompressionException.class)
	public void testInvalidJoinWithToManyUniqueValues() {
		AMapToData a = MapToFactory.create(10, 10000000);
		AMapToData b = MapToFactory.create(10, 10000000);
		MapToFactory.join(a, b);
	}

	@Test
	public void test_null_argument_01() {
		AMapToData a = null;
		AMapToData b = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData c = MapToFactory.join(a, b);
		compare(c, new int[] {1, 0, 0, 0, 0, 1, 1, 1, 1, 1});
		// compare(c, new int[] {0, 1, 1, 1, 1, 0, 0, 0, 0, 0});
	}

	@Test
	public void test_null_argument_02() {
		AMapToData a = MapToFactory.create(10, true, new IntArrayList[] {gen(new int[] {1, 2, 3, 4})});
		AMapToData b = null;
		AMapToData c = MapToFactory.join(a, b);
		compare(c, new int[] {1, 0, 0, 0, 0, 1, 1, 1, 1, 1});
	}

	@Test
	public void construct_with_zeros_false() {
		AMapToData a = MapToFactory.create(10, false,
			new IntArrayList[] {gen(new int[] {0, 1, 2, 3, 4}), gen(new int[] {5, 6, 7, 8, 9})});
		compare(a, new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
	}

	private static void compare(AMapToData res, int[] expected) {
		StringBuilder sb = new StringBuilder();
		sb.append("\nExpected:\n");
		sb.append(Arrays.toString(expected));
		sb.append("\nActual:\n");
		sb.append(res.toString());
		sb.append("\n");
		for(int i = 0; i < expected.length; i++) {
			assertEquals(sb.toString(), expected[i], res.getIndex(i));
		}
	}

	private static IntArrayList gen(int[] in) {
		return new IntArrayList(in);
	}
}
