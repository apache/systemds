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

package org.apache.sysds.test.component.compress.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.ACount.DArrCounts;
import org.apache.sysds.runtime.compress.utils.ACount.DCounts;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.junit.Test;

public class CountTest {
	@Test
	public void sort1() {
		DCounts a = new DCounts(1.0, 0, 1);
		DCounts h = a;
		a.next = new DCounts(2.0, 2, 3);
		a = a.next();
		a.next = new DCounts(2.0, 1, 4);
		a = a.next();

		try {
			DCounts hs = h.sort();
			
			assertEquals(4, hs.count);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void get1() {
		DCounts a = new DCounts(1.0, 0, 1);
		DCounts h = a;
		a.next = new DCounts(2.0, 2, 3);
		a = a.next();
		a.next = new DCounts(3.0, 1, 4);
		a = a.next();

		assertEquals(2, h.get(2.0).id);
		assertEquals(3, h.get(2.0).count);
		assertEquals(1, h.get(3.0).id);
		assertEquals(4, h.get(3.0).count);
		assertEquals(0, h.get(1.0).id);
		assertEquals(1, h.get(1.0).count);
		assertEquals(null, h.get(4.0));
	}

	@Test
	public void inc() {
		DCounts a = new DCounts(1.0, 0, 1);
		DCounts h = a;
		a.next = new DCounts(2.0, 2, 3);
		a = a.next();
		a.next = new DCounts(3.0, 1, 4);
		a = a.next();

		assertEquals(2, h.inc(2.0, 3, 13).id);
		assertEquals(1, h.inc(3.0, 3, 13).id);
		assertEquals(0, h.inc(1.0, 3, 13).id);
		assertEquals(null, h.get(4.0));
		assertNotEquals(null, h.inc(4.0, 3, 13));
		assertEquals(13, h.get(4.0).id);
		assertEquals(3, h.get(4.0).count);
		assertEquals(3 + 3, h.get(2.0).count);
		assertEquals(4 + 3, h.get(3.0).count);
		assertEquals(1 + 3, h.get(1.0).count);
	}

	@Test(expected = NotImplementedException.class)
	public void getDouble() {
		ACount<DblArray> a = new DArrCounts(new DblArray(new double[] {1, 2}), 1);
		a.get(0.0);
	}

	@Test(expected = NotImplementedException.class)
	public void incDouble() {
		ACount<DblArray> a = new DArrCounts(new DblArray(new double[] {1, 2}), 1);
		a.inc(0.0, 0, 1);
	}
}
