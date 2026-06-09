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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.sysds.runtime.compress.utils.HashMapIntToInt;
import org.junit.Test;

public class HashMapIntToIntTest {

	@Test
	public void basic1() {
		basic(new HashMapIntToInt(1));
	}

	@Test
	public void basic2() {
		basic(new HashMapIntToInt(2));
	}

	@Test
	public void basic16() {
		basic(new HashMapIntToInt(16));
	}

	@Test
	public void basic100() {
		basic(new HashMapIntToInt(100));
	}

	private void basic(HashMapIntToInt a) {
		assertTrue(a.isEmpty());
		assertEquals(0, a.size());

		// first insert via putIfAbsentI returns the absent sentinel -1
		assertEquals(-1, a.putIfAbsentI(1, 10));
		assertFalse(a.isEmpty());
		// second insert of same key keeps the existing value and returns it
		assertEquals(10, a.putIfAbsentI(1, 99));
		assertEquals(1, a.size());

		for(int i = 2; i < 10; i++)
			assertEquals(-1, a.putIfAbsentI(i, i * 10));
		assertEquals(9, a.size());

		// lookups
		assertEquals(10, a.getI(1));
		assertEquals(90, a.getI(9));
		assertEquals(-1, a.getI(13)); // absent
		assertEquals(Integer.valueOf(40), a.get(4));
		assertNull(a.get(13)); // absent via boxed accessor

		// containment
		assertTrue(a.containsKey(4));
		assertFalse(a.containsKey(42));
		assertTrue(a.containsValue(40));
		assertFalse(a.containsValue(41));

		// iteration covers exactly the inserted entries
		Set<Integer> keys = new HashSet<>();
		Set<Integer> vals = new HashSet<>();
		for(Entry<Integer, Integer> e : a.entrySet()) {
			keys.add(e.getKey());
			vals.add(e.getValue());
		}
		assertEquals(9, keys.size());
		for(int i = 1; i < 10; i++) {
			assertTrue(keys.contains(i));
			assertTrue(vals.contains(i * 10));
		}
	}

	@Test
	public void putReturnsPreviousValue() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		// putI returns -1 when the key is new
		assertEquals(-1, a.putI(5, 50));
		// putI overwrites and returns the previous value
		assertEquals(50, a.putI(5, 51));
		assertEquals(51, a.getI(5));
		assertEquals(1, a.size()); // overwrite does not grow the map
	}

	@Test
	public void putBoxedReturnsNullThenPrevious() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		assertNull(a.put(7, 70));
		assertEquals(Integer.valueOf(70), a.put(7, 71));
		assertEquals(Integer.valueOf(71), a.get(7));
	}

	@Test
	public void putIfAbsentDoesNotOverwrite() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		assertEquals(-1, a.putIfAbsentI(3, 30));
		assertEquals(30, a.putIfAbsentI(3, 31)); // returns existing, keeps 30
		assertEquals(30, a.getI(3));
		assertNull(a.putIfAbsent(8, 80));
		assertEquals(Integer.valueOf(80), a.putIfAbsent(8, 81));
		assertEquals(80, a.getI(8));
	}

	@Test
	public void putIfAbsentReturnValSemantics() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		// when absent: inserts and returns the newly stored value
		assertEquals(40, a.putIfAbsentReturnVal(4, 40));
		// when present: returns the existing value, does not overwrite
		assertEquals(40, a.putIfAbsentReturnVal(4, 99));
		assertEquals(40, a.getI(4));

		// the *Hash variant has identical semantics
		assertEquals(50, a.putIfAbsentReturnValHash(5, 50));
		assertEquals(50, a.putIfAbsentReturnValHash(5, 99));
		assertEquals(50, a.getI(5));
	}

	@Test
	public void absentSignaledByMinusOneSentinel() {
		// Design contract: the primitive int accessors (getI / putI / putIfAbsentI)
		// signal "absent" or "no previous value" with the sentinel -1 instead of a
		// nullable Integer. This is a deliberate performance choice to avoid boxing
		// and null handling on the hot path, so the tests pin the -1 behavior down.
		HashMapIntToInt a = new HashMapIntToInt(16);

		// lookup of an absent key returns the sentinel (not null, no exception)
		assertEquals(-1, a.getI(1));
		assertEquals(-1, a.getI(Integer.MAX_VALUE));

		// inserting a previously-absent key returns the sentinel (no prior value)
		assertEquals(-1, a.putI(1, 100));
		assertEquals(-1, a.putIfAbsentI(2, 200));

		// a populated map still returns the sentinel for any missing key
		assertEquals(-1, a.getI(3));

		// Consequence of the sentinel: -1 is reserved and must not be stored as a
		// value. A stored -1 is indistinguishable from "absent" through both the
		// primitive and the boxed accessors, which callers are required to respect.
		HashMapIntToInt b = new HashMapIntToInt(16);
		b.putI(7, -1);
		assertEquals(1, b.size()); // the entry really is stored
		assertEquals(-1, b.getI(7)); // ...but reads back as the absent sentinel
		assertNull(b.get(7)); // and the boxed accessor reports null as well
	}

	@Test
	public void resizeRetainsAllEntries() {
		// start small to force several resizes (load factor 0.75)
		HashMapIntToInt a = new HashMapIntToInt(1);
		final int n = 1000;
		for(int i = 0; i < n; i++)
			assertEquals(-1, a.putI(i, i * 2));
		assertEquals(n, a.size());
		for(int i = 0; i < n; i++)
			assertEquals(i * 2, a.getI(i));
		assertEquals(-1, a.getI(n)); // still absent after resizing

		// iteration still sees every entry after resizing
		Set<Integer> keys = new HashSet<>();
		for(Entry<Integer, Integer> e : a.entrySet())
			keys.add(e.getKey());
		assertEquals(n, keys.size());
	}

	@Test
	public void negativeAndBoundaryKeys() {
		HashMapIntToInt a = new HashMapIntToInt(8);
		int[] keys = {-1000, -1, 0, 1, Integer.MIN_VALUE, Integer.MAX_VALUE};
		for(int i = 0; i < keys.length; i++)
			a.putI(keys[i], i + 100);
		assertEquals(keys.length, a.size());
		for(int i = 0; i < keys.length; i++) {
			assertEquals(i + 100, a.getI(keys[i]));
			assertTrue(a.containsKey(keys[i]));
		}
	}

	@Test
	public void forEachVisitsAllEntries() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		for(int i = 0; i < 50; i++)
			a.putI(i, i + 1);
		int[] count = new int[] {0};
		long[] sum = new long[] {0};
		a.forEach((k, v) -> {
			count[0]++;
			sum[0] += (v - k); // each entry contributes exactly 1
		});
		assertEquals(50, count[0]);
		assertEquals(50, sum[0]);
	}

	@Test
	public void collisionChainsPutAndGet() {
		// capacity 16 -> 16 buckets; staying <= 12 entries avoids a resize, so
		// keys that are congruent mod 16 deterministically share one bucket.
		HashMapIntToInt a = new HashMapIntToInt(16);
		assertEquals(-1, a.putI(1, 100));
		assertEquals(-1, a.putI(17, 200)); // appended as 2nd node in the chain
		assertEquals(-1, a.putI(33, 300)); // traverses node1 -> node2, then appends
		assertEquals(3, a.size());

		// overwrite a node deep in the chain returns the previous value
		assertEquals(300, a.putI(33, 333));
		assertEquals(333, a.getI(33)); // getI walks the chain to the last node
		// a miss whose key maps to a populated bucket walks the chain, then -1
		assertEquals(-1, a.getI(49));

		// iterating a multi-node bucket exercises the iterator chain advance
		int cnt = 0;
		int sum = 0;
		for(Entry<Integer, Integer> e : a.entrySet()) {
			cnt++;
			sum += e.getValue();
		}
		assertEquals(3, cnt);
		assertEquals(100 + 200 + 333, sum);
	}

	@Test
	public void collisionChainsPutIfAbsent() {
		HashMapIntToInt a = new HashMapIntToInt(16);
		// putIfAbsentI into a shared bucket: create, then append into the chain
		assertEquals(-1, a.putIfAbsentI(1, 10));
		assertEquals(-1, a.putIfAbsentI(17, 20));
		assertEquals(10, a.putIfAbsentI(1, 99)); // match first node, keep 10
		assertEquals(20, a.putIfAbsentI(17, 99)); // match deeper node, keep 20

		// the *ReturnVal variants append into the same non-empty bucket
		assertEquals(30, a.putIfAbsentReturnVal(33, 30)); // appended -> new value
		assertEquals(30, a.putIfAbsentReturnVal(33, 99)); // present -> existing
		assertEquals(40, a.putIfAbsentReturnValHash(49, 40)); // appended -> new value
		assertEquals(40, a.putIfAbsentReturnValHash(49, 99)); // present -> existing
		assertEquals(4, a.size());
	}

	@Test
	public void toStringContainsEntries() {
		HashMapIntToInt a = new HashMapIntToInt(16);
		a.putI(2, 20);
		a.putI(3, 30);
		String s = a.toString();
		assertTrue(s.contains("(2\u219220)")); // (2->20)
		assertTrue(s.contains("(3\u219230)")); // (3->30)
	}

	@Test
	public void entrySetSize() {
		HashMapIntToInt a = new HashMapIntToInt(16);
		for(int i = 0; i < 5; i++)
			a.putI(i, i);
		assertEquals(5, a.entrySet().size());
	}

	@Test
	public void resizeWithEmptyAndChainedBuckets() {
		// 16 buckets, load factor 0.75 -> a resize fires once size exceeds 12.
		// Pre-seed a colliding chain in bucket 1 (keys congruent mod 16), then add
		// distinct keys so that at resize time oldBuckets holds both a multi-node
		// chain (if(n != null) true + chain re-put) and empty buckets (false side).
		HashMapIntToInt a = new HashMapIntToInt(16);
		a.putI(1, 1);
		a.putI(17, 17); // chain node in bucket 1
		for(int i = 2; i <= 12; i++) // distinct buckets, drives size past 12
			a.putI(i, i);
		assertEquals(13, a.size()); // triggers exactly one resize

		assertEquals(1, a.getI(1));
		assertEquals(17, a.getI(17));
		for(int i = 2; i <= 12; i++)
			assertEquals(i, a.getI(i));
	}

	@Test
	public void resizeWithEmptyBucketsInOldTable() {
		// A resize only fires from the collision-append path, so dense sequential
		// keys fill every bucket before the load factor is exceeded and the old
		// table is always full at rehash time. Here we instead pile keys that are
		// all congruent mod 16 into a single chain: the load factor (0.75) is
		// crossed while 15 of the 16 buckets stay empty, exercising the
		// n == null (skip empty bucket) branch of resize()'s rehash loop.
		HashMapIntToInt a = new HashMapIntToInt(16);
		final int n = 13; // 13 > 16 * 0.75 triggers exactly one resize
		for(int i = 0; i < n; i++)
			assertEquals(-1, a.putI(i * 16, i)); // every key maps to bucket 0 at capacity 16
		assertEquals(n, a.size());

		// every entry survived the rehash over a table that contained empty buckets
		for(int i = 0; i < n; i++)
			assertEquals(i, a.getI(i * 16));
	}

	@Test
	public void emptyEntrySetIteration() {
		HashMapIntToInt a = new HashMapIntToInt(16);
		int cnt = 0;
		for(Entry<Integer, Integer> e : a.entrySet())
			cnt += e.getValue();
		assertEquals(0, cnt);
		assertFalse(a.entrySet().iterator().hasNext());
		assertEquals(0, a.entrySet().size());
	}

	@Test
	public void forEachOverChain() {
		// colliding keys (congruent mod 16) build a multi-node bucket so forEach
		// walks the linked list within a bucket
		HashMapIntToInt a = new HashMapIntToInt(16);
		a.putI(1, 1);
		a.putI(17, 1);
		a.putI(33, 1);
		int[] count = new int[] {0};
		a.forEach((k, v) -> count[0]++);
		assertEquals(3, count[0]);
	}

	@Test
	public void containsValueNonInteger() {
		HashMapIntToInt a = new HashMapIntToInt(4);
		a.putI(1, 1);
		assertFalse(a.containsValue("not-an-integer"));
		assertFalse(a.containsValue(null));
	}

	@Test(expected = UnsupportedOperationException.class)
	public void putAllUnsupported() {
		new HashMapIntToInt(4).putAll(new java.util.HashMap<Integer, Integer>());
	}

	@Test(expected = UnsupportedOperationException.class)
	public void removeUnsupported() {
		new HashMapIntToInt(4).remove(1);
	}

	@Test(expected = UnsupportedOperationException.class)
	public void clearUnsupported() {
		new HashMapIntToInt(4).clear();
	}

	@Test(expected = UnsupportedOperationException.class)
	public void keySetUnsupported() {
		new HashMapIntToInt(4).keySet();
	}

	@Test(expected = UnsupportedOperationException.class)
	public void valuesUnsupported() {
		new HashMapIntToInt(4).values();
	}

	@Test
	public void randomKeysMatchReference1() {
		randomKeysMatchReference(1, 1, 2000);
	}

	@Test
	public void randomKeysMatchReference16() {
		randomKeysMatchReference(2, 16, 2000);
	}

	@Test
	public void randomKeysMatchReferenceSmallDomain() {
		// a small key domain relative to the entry count forces many overwrites
		// and unevenly loaded buckets rather than a clean one-key-per-bucket layout
		randomKeysMatchReference(3, 4, 3000);
	}

	private void randomKeysMatchReference(long seed, int capacity, int inserts) {
		// Cross-check against java.util.HashMap under randomized, value-shifted keys
		// so bucket load is genuinely uneven (collisions, chains, and empty buckets
		// coexist) instead of the perfectly balanced layout that dense keys produce.
		// Values are kept >= 0 because the primitive accessors reserve -1 as the
		// "absent" sentinel.
		Random r = new Random(seed);
		HashMapIntToInt a = new HashMapIntToInt(capacity);
		Map<Integer, Integer> ref = new HashMap<>();

		for(int i = 0; i < inserts; i++) {
			int key = r.nextInt(inserts); // domain may be smaller than #inserts -> overwrites
			int value = r.nextInt(Integer.MAX_VALUE); // never -1
			Integer prev = ref.put(key, value);
			int prevI = a.putI(key, value);
			if(prev == null)
				assertEquals(-1, prevI); // first time we see the key
			else
				assertEquals(prev.intValue(), prevI); // overwrite returns previous value
		}

		assertEquals(ref.size(), a.size());
		for(Entry<Integer, Integer> e : ref.entrySet()) {
			assertEquals(e.getValue().intValue(), a.getI(e.getKey()));
			assertTrue(a.containsKey(e.getKey()));
		}

		// iteration visits exactly the reference entries, nothing more or less
		Map<Integer, Integer> seen = new HashMap<>();
		for(Entry<Integer, Integer> e : a.entrySet())
			assertNull("duplicate key from iterator: " + e.getKey(), seen.put(e.getKey(), e.getValue()));
		assertEquals(ref, seen);

		// a few keys outside the inserted domain must report absent
		for(int i = 0; i < 50; i++)
			assertEquals(-1, a.getI(inserts + r.nextInt(inserts) + 1));
	}

	@Test
	public void randomPutIfAbsentKeepsFirstValue() {
		// putIfAbsentI must preserve the first value stored for a key even under
		// randomized, colliding inserts; mirror that contract with a reference map.
		Random r = new Random(7);
		HashMapIntToInt a = new HashMapIntToInt(2);
		Map<Integer, Integer> ref = new HashMap<>();
		final int inserts = 2000;

		for(int i = 0; i < inserts; i++) {
			int key = r.nextInt(inserts / 4); // small domain -> frequent collisions
			int value = r.nextInt(Integer.MAX_VALUE); // never -1
			if(ref.containsKey(key))
				assertEquals(ref.get(key).intValue(), a.putIfAbsentI(key, value)); // keep first
			else {
				assertEquals(-1, a.putIfAbsentI(key, value));
				ref.put(key, value);
			}
		}

		assertEquals(ref.size(), a.size());
		for(Entry<Integer, Integer> e : ref.entrySet())
			assertEquals(e.getValue().intValue(), a.getI(e.getKey()));
	}
}
