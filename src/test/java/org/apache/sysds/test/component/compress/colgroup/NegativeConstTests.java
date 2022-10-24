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

package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.junit.Test;

public class NegativeConstTests {

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_01() {
		ColGroupConst.create(-1, 14);
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_02() {
		ColGroupConst.create(0, 14);
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_03() {
		ColGroupConst.create(new int[] {}, 0);
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_05() {
		ColGroupConst.create(new int[] {0, 1, 2}, new double[] {1, 2});
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_06() {
		ColGroupConst.create(new int[] {0, 1}, new double[] {1, 2, 4});
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_07() {
		ColGroupConst.create(2, Dictionary.createNoCheck(new double[] {1, 2, 4}));
	}

	@Test(expected = DMLCompressionException.class)
	public void testConstConstruction_08() {
		ColGroupConst.create(4, Dictionary.createNoCheck(new double[] {1, 2, 4}));
	}

	@Test
	public void testConstConstruction_allowed_01() {
		ColGroupConst.create(new int[] {0, 1, 2, 3}, 0);
	}

	@Test
	public void testConstConstruction_allowed_02() {
		ColGroupConst.create(3, Dictionary.createNoCheck(new double[] {1, 2, 4}));
	}

	@Test
	public void testConstConstruction_allowed_03() {
		ColGroupConst.create(new double[] {1, 2, 4});
	}

	@Test(expected = NullPointerException.class)
	public void testConstConstruction_null_01() {
		ColGroupConst.create(null, 0);
	}

	@Test(expected = NullPointerException.class)
	public void testConstConstruction_null_02() {
		ColGroupConst.create(null, null);
	}

	@Test(expected = NullPointerException.class)
	public void testConstConstruction_null_03() {
		ColGroupConst.create(0, null);
	}

	@Test(expected = NullPointerException.class)
	public void testConstConstruction_null_04() {
		ColGroupConst.create(null);
	}
}
