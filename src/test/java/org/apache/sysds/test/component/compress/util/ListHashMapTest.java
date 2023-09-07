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
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class ListHashMapTest {

	@Test
	public void add() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray a = new DblArray(new double[] {1, 2, 3});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			m.appendValue(a, 0);
		}
		IntArrayList l = m.get(a);
		assertEquals(rep, l.size());

	}

	@Test
	public void add2() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray a = new DblArray(new double[] {1, 2, 3});
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			m.appendValue(a, i);
			m.appendValue(b, i);
		}
		IntArrayList l = m.get(a);
		assertEquals(rep, l.size());

	}

	@Test
	public void add3() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			m.appendValue(a, i);
			m.appendValue(b, i);
		}
		IntArrayList l = m.get(b);
		assertEquals(rep, l.size());

	}

	@Test
	public void add4() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			m.appendValue(a, i);
		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
		}
		IntArrayList l = m.get(b);
		assertEquals(rep, l.size());

	}

	@Test
	public void extractAll() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			m.appendValue(a, i);
		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
		}

		assertEquals(rep + 1, m.extractValues().size());

	}

	@Test
	public void toStringWorks() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			m.appendValue(a, i);
		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
		}
		m.toString();
	}

	@Test
	public void size() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			m.appendValue(a, i);
			assertEquals(i + 1, m.size());
		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
			assertEquals(rep + 1, m.size());
		}
		m.toString();
	}

	@Test
	public void get() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap();
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			assertTrue(m.get(a) == null);
			m.appendValue(a, i);
			assertEquals(i + 1, m.size());
			assertTrue(m.get(a) != null);

		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
			assertEquals(rep + 1, m.size());
		}
		m.toString();
	}

	@Test
	public void getCustom() {
		DblArrayIntListHashMap m = new DblArrayIntListHashMap(25);
		DblArray b = new DblArray(new double[] {1, 2, 4});
		final int rep = 100;
		for(int i = 0; i < rep; i++) {
			DblArray a = new DblArray(new double[] {1, i, i});
			assertTrue(m.get(a) == null);
			m.appendValue(a, i);
			assertEquals(i + 1, m.size());
			assertTrue(m.get(a) != null);

		}
		for(int i = 0; i < rep; i++) {
			m.appendValue(b, i);
			assertEquals(rep + 1, m.size());
		}

	}
}
