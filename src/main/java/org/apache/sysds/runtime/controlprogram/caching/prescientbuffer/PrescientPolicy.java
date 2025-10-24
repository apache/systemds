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

import java.util.HashMap;
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
	 * Called by UMM's makeSpace() to decide which block to evict.
	 * * @param cache The set of all block IDs currently in the buffer
	 * @param pinned The list of all block IDs that are pinned
	 * @param currentTime The current logical time
	 * @return The block ID to evict
	 */
	public String evict(Set<String> cache, List<String> pinned, long currentTime) {
		// TODO: Implement "evict-furthest-in-future" logic here
		// 1. Iterate through every 'blockID' in 'cache'
		// 2. If 'blockID' is in 'pinned', ignore it.
		// 3. Use '_trace.getAccessTime(blockID)' to find its next access time > currentTime
		// 4. The block with the (largest next access time) or (no future access) is the winner.
		// 5. Return the winner's blockID.

		return null; // Placeholder
	}

	/**
	 * Called by UMM's prefetch() to decide which blocks to load.
	 * * @param currentTime The current logical time
	 * @return A list of block IDs to prefetch
	 */
	public List<String> getBlocksToPrefetch(long currentTime) {
		// TODO: Implement prefetch logic here
		// 1. Define a "prefetch window" (e.g., time T+1 to T+5)
		// 2. Iterate through all blocks in '_trace.getTrace()'
		// 3. Check if a block has an access time within that window
		// 4. If yes, add it to a list.
		// 5. Return the list of blocks.

		return java.util.Collections.emptyList(); // Placeholder
	}

	public void setTrace(IOTrace ioTrace) {
		_trace = ioTrace;
	}
}
