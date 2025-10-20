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
import java.util.Map;
import java.util.Set;

/**
 * Implement prescient buffer
 */
public class PrescientPolicy implements EvictionPolicy {

	// Map of block ID, access times
	private final Map<String, Long> accessTimeMap = new HashMap<>();

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

}
