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

package org.apache.sysds.runtime.controlprogram.caching.prescientbuffer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import java.util.List;
import java.util.Set;

public class PrescientPolicyTest {

	private IOTrace trace;
	private PrescientPolicy policy;

	/**
	 * Creates a mock IOTrace for all tests.
	 * * Access Pattern:
	 * T=1:  A
	 * T=2:  B
	 * T=3:  A
	 * T=4:  C
	 * T=5:  B
	 * T=6:  D  (D is never used again)
	 * T=7:  A
	 * T=8:  C
	 * T=9:  E  (E will be pinned)
	 * T=10: B
	 */
	@Before
	public void setUp() {
		trace = new IOTrace();
		// Access times are automatically sorted by IOTrace
		trace.recordAccess("A", 1);
		trace.recordAccess("B", 2);
		trace.recordAccess("A", 3);
		trace.recordAccess("C", 4);
		trace.recordAccess("B", 5);
		trace.recordAccess("D", 6); // D is never used again after this
		trace.recordAccess("A", 7);
		trace.recordAccess("C", 8);
		trace.recordAccess("E", 9); // E will be our pinned block
		trace.recordAccess("B", 10);

		policy = new PrescientPolicy();
		policy.setTrace(trace);
	}

	@Test
	public void testGetBlocksToPrefetchAtStart() {
		// Window is 5. At T=0, we look at (1, 2, 3, 4, 5]
		long currentTime = 0;
		List<String> blocks = policy.getBlocksToPrefetch(currentTime);

		// Should prefetch A (at 1), B (at 2), C (at 4)
		assertEquals(3, blocks.size());
		assertTrue(blocks.containsAll(List.of("A", "B", "C")));
	}

	@Test
	public void testGetBlocksToPrefetchInMiddle() {
		// At T=5, we look at (6, 7, 8, 9, 10]
		long currentTime = 5;
		List<String> blocks = policy.getBlocksToPrefetch(currentTime);

		// Should prefetch D (at 6), A (at 7), C (at 8), E (at 9), B (at 10)
		assertEquals(5, blocks.size());
		assertTrue(blocks.containsAll(List.of("A", "B", "C", "D", "E")));
	}

	@Test
	public void testGetBlocksToPrefetchAtEnd() {
		// At T=10, we look at (11, 12, 13, 14, 15]
		long currentTime = 10;
		List<String> blocks = policy.getBlocksToPrefetch(currentTime);

		// No blocks accessed in this window
		assertEquals(0, blocks.size());
		assertTrue(blocks.isEmpty());
	}

	@Test
	public void testEvictFindsNeverUsedBlock() {
		// Cache has A, B, C, D. E is pinned (not in this cache set).
		// Time T=6 (just after D was used)
		long currentTime = 6;
		Set<String> cache = Set.of("A", "B", "C", "D");
		List<String> pinned = List.of();

		// Next uses:
		// A: at T=7
		// B: at T=10
		// C: at T=8
		// D: never (Long.MAX_VALUE)

		String evictCandidate = policy.evict(cache, pinned, currentTime);
		// Policy should immediately evict D
		assertEquals("D", evictCandidate);
	}

	@Test
	public void testEvictFindsFurthestInFuture() {
		// Cache has A, B, C. D was already evicted. E is pinned.
		long currentTime = 6;
		Set<String> cache = Set.of("A", "B", "C", "E");
		List<String> pinned = List.of("E"); // E cannot be evicted

		// Next uses:
		// A: at T=7
		// B: at T=10
		// C: at T=8
		// E: at T=9 (but pinned)

		String evictCandidate = policy.evict(cache, pinned, currentTime);
		// Policy should evict B (used at T=10, furthest away)
		assertEquals("B", evictCandidate);
	}

	@Test
	public void testEvictAllPinned() {
		// All blocks in cache are pinned
		long currentTime = 0;
		Set<String> cache = Set.of("A", "E");
		List<String> pinned = List.of("A", "E");

		String evictCandidate = policy.evict(cache, pinned, currentTime);
		// Should return null (no valid eviction candidate)
		assertNull(evictCandidate);
	}

	//	@Test
//	public void testBasicEviction() {
//		PrescientPolicy policy = new PrescientPolicy();
//
//		policy.setAccessTime("block1", 10);
//		policy.setAccessTime("block2", 40);
//		policy.setAccessTime("block3", 25);
//
//		Set<String> candidates = new HashSet<>();
//		assertNull(policy.selectBlockForEviction(candidates));
//
//		candidates.add("block1");
//		candidates.add("block2");
//		candidates.add("block3");
//		assertEquals("block2", policy.selectBlockForEviction(candidates));
//
//	}
}
