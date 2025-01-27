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
package org.apache.sysds.test.component.frame.array;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.sysds.runtime.frame.data.columns.HashMapToInt;
import org.junit.Test;

public class HashMapToIntTest {

	@Test
	public void insert() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		m.put(1, 1);
		assertTrue(m.containsKey(1));
		assertTrue(m.containsValue(1));
	}

	@Test
	public void isEmpty() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		assertTrue(m.isEmpty());
		m.put(1, 1);
		assertFalse(m.isEmpty());
	}

	@Test
	public void insert10() {

		Map<Integer, Integer> m = new HashMapToInt<>(10);
		for(int i = 0; i < 10; i++) {
			m.put(i, i);
			assertFalse(m.isEmpty());
			assertTrue(m.containsKey(i));
			assertTrue(m.containsValue(i));
		}

		for(int i = 0; i < 10; i++) {
			assertTrue(m.containsKey(i));
			assertTrue(m.containsValue(i));
		}
	}

	@Test
	public void forEach() {

		Map<Integer, Integer> m = new HashMapToInt<>(10);
		Map<Integer, Integer> m2 = new HashMap<>();
		Random r = new Random(32);
		for(int i = 0; i < 100; i++) {
			int v1 = r.nextInt();
			int v2 = r.nextInt();
			m.put(v1, v2);
			m2.put(v1, v2);
		}

		assertEquals(m.size(), m2.size());
		for(Entry<Integer, Integer> e : m2.entrySet()) {
			assertTrue(m.containsKey(e.getKey()));
		}

		assertEquals(m.size(), m2.size());
		for(Entry<Integer, Integer> e : m.entrySet()) {
			assertTrue(m2.containsKey(e.getKey()));
			assertEquals(m.get(e.getKey()), m2.get(e.getKey()));
		}

	}

	@Test
	public void doNotContainKey() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		for(int i = 0; i < 100; i++) {
			assertFalse(m.containsKey(i));
			assertFalse(m.containsValue(i * 10000));
			m.put(i, i * 10000);
			assertTrue(m.containsKey(i));
			assertTrue(m.containsValue(i * 10000));
			assertEquals(m.get(i), Integer.valueOf(i * 10000));
		}

	}

	@Test
	public void doNotContainValue() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);

		assertFalse(m.containsValue(new Object()));

	}

	@Test
	public void overwriteKey() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);

		Integer v;
		v = m.put(1, 10);
		assertEquals(Integer.valueOf(10), m.get(Integer.valueOf(1)));
		assertEquals(v, null);
		v = m.put(1, 11);
		assertEquals(v, Integer.valueOf(10));
		assertEquals(Integer.valueOf(11), m.get(Integer.valueOf(1)));
		v = m.put(1, 12);
		assertEquals(v, Integer.valueOf(11));
		assertEquals(Integer.valueOf(12), m.get(Integer.valueOf(1)));
	}

	@Test
	public void forEach2() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		for(int i = 900; i < 1000; i++) {
			m.put(i, i * 32121523);
		}
		final Map<Integer, Integer> m2 = new HashMap<>();
		m.forEach((k, v) -> m2.put(k, v));
		m2.forEach((k, v) -> assertTrue("key missing: " + k, m.containsKey(k)));
	}

	@Test
	public void testToString() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);

		assertEquals(0, m.toString().length());
		m.put(555, 321);
		String s = m.toString();
		assertTrue(s.contains("555"));
		assertTrue(s.contains("321"));
	}

	@Test
	public void testSizeOfKeySet() {
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		for(int i = 0; i < 10; i++) {
			m.put(i * 321, i * 3222);
			assertEquals(m.size(), m.entrySet().size());
		}
	}

	@Test 
	public void putIfAbsent(){
		Map<Integer, Integer> m = new HashMapToInt<>(10);
		for(int i = 0; i < 1000; i++) {
			assertNull(m.putIfAbsent(i * 321, i * 3222));

		}

		for(int i = 0; i < 1000; i++) {
			assertEquals(i*3222,(int)m.putIfAbsent(i * 321, i * 3222));
		}
	}
}
