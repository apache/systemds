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

package org.apache.sysds.runtime.controlprogram.caching;

import java.util.List;
import java.util.Set;

/**
 * An interface all Buffer pool eviction policies,
 * for pluggable eviction strategies - LRU, FIFO, Prescient
 */
public interface EvictionPolicy {

//	/**
//	 * Finds the unpinned block that won't be used in near future (or never used)
//	 * to evict from the cache.
//	 *
//	 * @param cache The set of all block IDs currently in the buffer
//	 * @param pinned The list of all block IDs that are pinned
//	 * @param currentTime The current logical time
//	 * @return The block ID used for eviction, or null if all blocks are pinned.
//	 */
//	public String evict(Set<String> cache, List<String> pinned, long currentTime);

}
