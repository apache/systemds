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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implement prescient buffer
 */
public class PrescientPolicy implements EvictionPolicy {

	// Map of block ID, access times
	private final Map<String, Long> accessTimeMap = new HashMap<>();
	private IOTrace _trace;
	// Defines how many logical time units to look ahead for prefetching
	private static final int PREFETCH_WINDOW = 5;

	// register blocks with access time
	public void setAccessTime(String blockId, long accessTime) {
		accessTimeMap.put(blockId, accessTime);
	}

	/**
	 * Select a block to evict from the given list of candidates
	 *
	 * @param candidates A set of candidate block identifiers for currently in buffer
	 * @return The identifier of the block chosen for eviction
	 */
	@Override
	public String selectBlockForEviction(Set<String> candidates) {
		// base case
		if (candidates == null || candidates.isEmpty()) {
			return null;
		}

		String selected = null;
		long maxTime = -1;

		for (String candidate : candidates) {
			long time = accessTimeMap.getOrDefault(candidate, Long.MAX_VALUE);

			if (time > maxTime) {
				maxTime = time;
				selected = candidate;
			}
		}

		return selected;
	}

	/**
	 * Finds the next time a block is accessed, <b>after</b> the current time.
	 *
	 * @param blockID The block to check
	 * @param currentTime The current logical time
	 * @return The logical time of the next access, or Long.MAX_VALUE if never used again.
	 */
	private long findNextAccess(String blockID, long currentTime) {
		if (_trace == null) {
			return Long.MAX_VALUE;
		}

		List<Long> accessTimes = _trace.getAccessTime(blockID);
		// Find the first access time that is greater than the current time
		for (long time : accessTimes) {
			if (time > currentTime) {
				return time;
			}
		}

		// This block is never accessed again in the future
		return Long.MAX_VALUE;
	}

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
			long nextAccessTime = findNextAccess(blockID, currentTime);

			// find the block that's never used again
			if (nextAccessTime == Long.MAX_VALUE) {
				return blockID;
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

		// Use a Set to store unique block IDs
		Set<String> blocksToPrefetch = new HashSet<>();
		long lookaheadTime = currentTime + PREFETCH_WINDOW;

		// Iterate over all blocks in the trace
		for (String blockID : _trace.getTrace().keySet()) {
			List<Long> accessTimes = _trace.getAccessTime(blockID);

			// Check if this block is accessed within our prefetch window
			for (long time : accessTimes) {
				if (time > currentTime && time <= lookaheadTime) {
					blocksToPrefetch.add(blockID);
				}
			}
		}

		return new ArrayList<>(blocksToPrefetch);
	}

	public void setTrace(IOTrace ioTrace) {
		_trace = ioTrace;
	}
}
