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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.junit.Test;

public class SingletonLookupHashMapTest {

	@Test
	public void testCreate() {
		SingletonLookupHashMap hm1 = SingletonLookupHashMap.getMap();
		SingletonLookupHashMap hm2 = SingletonLookupHashMap.getMap();
		assertEquals(hm1, hm2);
	}

	@Test
	public void testPut() {
		String input = "Hi";
		int id = getM().put(input);
		String ret = (String) getM().get(id);
		assertEquals(input, ret);
	}

	@Test
	public void testToString() {
		// alittle meta .. but take the string of it and put it inside the map.
		String input = getM().toString();
		putAndGet(input);
	}

	public void putAndGet(Object o) {
		int id = getM().put(o);
		Object ret = getM().get(id);
		assertEquals(o, ret);
	}

	@Test
	public void containsID() {
		int id = getM().put("Hi 2");
		assertTrue(getM().containsKey(id));
	}

	@Test
	public void doesNotContainID() {
		int id = getM().put("Hi 4215");
		getM().removeKey(id);
		assertTrue(!getM().containsKey(id));
	}

	@Test
	public void removeKey() {
		int id = getM().put("Hi 12561");
		assertTrue(getM().containsKey(id));
		getM().removeKey(id);
		assertTrue(!getM().containsKey(id));
	}

	@Test
	public void removeNotExistingKey() {
		getM().removeKey(132241);
	}

	@Test
	public void putBadHash() {
		int id1 = getM().put(new SuperBadHashingObject());
		int id2 = getM().put(new SuperBadHashingObject());
		assertTrue(id1 != id2);
	}

	private static SingletonLookupHashMap getM() {
		return SingletonLookupHashMap.getMap();
	}

	static class SuperBadHashingObject {

		SuperBadHashingObject() {
		}

		@Override
		public int hashCode() {
			// bad hash ... but good for testing.
			return 0;
		}
	}
}
