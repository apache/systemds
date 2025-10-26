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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * IOTrace holds the pre-computed I/O access trace for the OOC operations.
 */
public class IOTrace {

	// Block ID vs unique accesses
	private final Map<String, LinkedList<Long>> _trace;

	private long _currentTime;

	public IOTrace() {
		_trace = new HashMap<>();
		_currentTime = 0;
	}

	/**
	 * Access to the block at a current time
	 */
	public void recordAccess(String blockID, long logicalTime) {
		_trace.computeIfAbsent(blockID, k -> new LinkedList<>()).add(logicalTime);
	}

	/**
	 * Get all access times for a given block
	 * @param blockID Block ID
	 * @return all the access times
	 */
	public LinkedList<Long> getAccessTime(String blockID) {
		return _trace.getOrDefault(blockID, new LinkedList<>());
	}

	/**
	 * Get the next access time for a block after currentTime
	 *
	 * @param blockID the block identifier
	 * @param currentTime current logical time
	 * @return next access time or Long.MAX_VALUE if never accessed again
	 */
	public long getNextAccessTime(String blockID, long currentTime) {
		LinkedList<Long> accesses = getAccessTime(blockID);
		if (accesses == null ||  accesses.isEmpty()) {
			return Long.MAX_VALUE; // won't access again
		}

		return accesses.peekFirst();
	}

	/**
	 * Get all the blocks in a prefetch window
	 *
	 * @param currentTime current logical time
	 * @param windowSize prefetch lookahead window
	 * @return List of block IDs to prefetch
	 */
	public List<String> getBlocksInWindow(long currentTime, long windowSize) {
		List<String> blocks = new ArrayList<>();

		for (Map.Entry<String, LinkedList<Long>> entry : _trace.entrySet()) {
			String  blockID = entry.getKey();
			long nextAccess = getNextAccessTime(blockID, currentTime);

			if ( nextAccess != Long.MAX_VALUE &&
							nextAccess > currentTime &&
							nextAccess <= currentTime + windowSize) {
				blocks.add(blockID);
			}
		}

		return blocks;
	}

	/**
	 * clean up the trace entries outside the sliding window
	 * this may or may not be required.
	 *
	 * @param currentTime current logical time
	 */
	public void cleanup(long currentTime) {
		_trace.entrySet().removeIf(entry -> {
			LinkedList<Long> accesses = _trace.get(entry.getKey());
			accesses.removeIf(time -> time <= currentTime);
			return accesses.isEmpty();
		});
	}

	/**
	 * Get the complete trace for debugging
	 * @return view of the _trace
	 */
	public Map<String, LinkedList<Long>> getTrace() {
		return _trace;
	}

	// clear all trace data
	public void clear() {
		_trace.clear();
		_currentTime = 0;
	}
}
