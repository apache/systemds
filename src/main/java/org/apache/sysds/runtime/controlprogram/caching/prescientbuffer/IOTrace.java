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
import java.util.List;
import java.util.Map;

/**
 * IOTrace holds the pre-computed I/O access trace for the OOC operations.
 */
public class IOTrace {

	// Block ID vs unique accesses
	private final Map<String, List<Long>> _trace;

	private long _currentTime;

	public IOTrace() {
		_trace = new HashMap<>();
		_currentTime = 0;
	}

	/**
	 * Access to the block at a current time
	 */
	public void recordAccess(String blockID, long logicalTime) {
		_trace.computeIfAbsent(blockID, k -> new ArrayList<>()).add(logicalTime);
	}

	/**
	 * Get all access times for a given block
	 * @param blockID Block ID
	 * @return all the access times
	 */
	public List<Long> getAccessTime(String blockID) {
		return _trace.getOrDefault(blockID, new ArrayList<>());
	}

	public Map<String, List<Long>> getTrace() {
		return _trace;
	}
}
