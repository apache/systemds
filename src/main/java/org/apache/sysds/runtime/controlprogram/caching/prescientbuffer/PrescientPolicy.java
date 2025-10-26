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

import org.apache.sysds.runtime.controlprogram.caching.EvictionPolicy;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Implement prescient buffer
 */
public class PrescientPolicy implements EvictionPolicy {

	private IOTrace _trace;
	// Defines how many logical time units to look ahead for prefetching
	private static final int PREFETCH_WINDOW = 5;

	/**
	 * Finds the unpinned block that won't be used in near future (or never used).
	 *
	 * @param cache The set of all block IDs currently in the buffer
	 * @param pinned The list of all block IDs that are pinned
	 * @param currentTime The current logical time
	 * @return The block ID used for eviction, or null if all blocks are pinned.
	 */
	public String evict(Set<String> cache, List<String> pinned, long currentTime) {
		if (cache == null || cache.isEmpty()) {
			return null;
		}

		String evictCandidate = null;
		long maxNextAccessTime = -1; // We're looking for the largest access time

		for (String blockID : cache) {
			// Cannot evict a pinned block
			if (pinned.contains(blockID)) {
				continue;
			}

			// Find the next time this block will be used
			long nextAccessTime = _trace.getNextAccessTime(blockID, currentTime);

			// case 1: find the block that's never used again
			if (nextAccessTime == Long.MAX_VALUE) {
				return blockID;
			}

			// case 2: find the block that's the furthest
			if (nextAccessTime > maxNextAccessTime) {
				maxNextAccessTime = nextAccessTime;
				evictCandidate = blockID;
			}

		}

		return evictCandidate;
	}

	/**
	 * Sliding Window implementation:
	 * Looks ahead N time units and finds all unique blocks accessed in that window.
	 *
	 * @param currentTime The current logical time
	 * @return A list of unique block IDs to prefetch
	 */
	public List<String> getBlocksToPrefetch(long currentTime) {
		if (_trace == null) {
			return Collections.emptyList();
		}

		return _trace.getBlocksInWindow(currentTime, PREFETCH_WINDOW);
	}

	public void setTrace(IOTrace ioTrace) {
		_trace = ioTrace;
	}
}
