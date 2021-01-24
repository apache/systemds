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

import org.apache.commons.lang.NotImplementedException;

/**
 * Unified Memory Manager - Initial Design
 * 
 * Motivation:
 * The Unified Memory Manager, henceforth UMM, will act as a central manager of in-memory
 * matrix (uncompressed and compressed), frame, and tensor blocks within SystemDS control
 * program. So far, operation memory (70%) and buffer pool memory (15%, LazyWriteBuffer)
 * are managed independently, which causes unnecessary evictions. New components like the
 * LineageCache also use and manage statically provisioned memory areas. Ultimately, the
 * UMM aims to eliminate these shortcomings by providing a central, potentially thread-local,
 * memory management.
 *
 * Memory Areas:
 * Initially, the UMM only handles CacheBlock objects (e.g., MatrixBlock, FrameBlock, and
 * TensorBlock), and manages two memory areas:
 *   (1) operation memory (pinned cache blocks and reserved memory) and
 *   (2) dirty objects (dirty cache blocks that need to be written to local FS before eviction)
 * 
 * The UMM is configured with a capacity (absolute size in byte). Relative to this capacity,
 * the operations and buffer pool memory areas each will have a min and max amount of memory
 * they can occupy, meaning that the boundary for the areas can shift dynamically depending
 * on the current load. Most importantly, though, dirty objects must not be counted twice
 * when pinning such an object for an operation. The min/max constraints are not exposed but
 * configured internally. An good starting point are the following constraints (relative to
 * JVM max heap size):
 * ___________________________
 * | operations  | 0%  | 70% | (pin requests always accepted)
 * | buffer pool | 15% | 85% | (eviction on demand)
 *
 * Object Lifecycle:
 * The UMM will also need to keep track of the current state of individual cache blocks, for
 * which it will have a few member variables. A queue similar to the current EvictionQueue is
 * used to add/remove entries with LRU as its eviction policy. In general, there are three
 * properties of object status to consider:
 *  (1) Non-dirty/dirty: non-dirty objects have a representation on HDFS or can be recomputed
 *      from lineage trace (e.g., rand/seq outputs), while dirty objects need to be preserved.
 *  (2) FS Persisted: on eviction, dirty objects need to be written to local file system.
 *      As long the local file representation exist, dirty objects can simply be dropped.
 *  (3) Pinned/unpinned: For operations, objects are pinned into memory to guard against
 *      eviction. All pin requests have to be accepted, and once a non-dirty object is released
 *      (unpinned) it can be dropped without persisting it to local FS.
 *
 * Thread-safeness:
 * Initially, the UMM will be used in an instance-based manner. For global visibility and
 * use in parallel for loops, the UMM would need to provide a static, synchronized API, but
 * this constitutes a source of severe contention. In the future, we will consider a design
 * with thread-local UMMs for the individual parfor workers.
 *
 * Testing:
 * The UMM will be developed bottom up, and thus initially tested via component tests for
 * evaluating the eviction behavior for sequences of API requests. 
 */
public class UnifiedMemoryManager
{
	public UnifiedMemoryManager(long capacity) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Pins a cache block into operation memory.
	 * 
	 * @param key    unique identifier and local FS filename for eviction
	 * @param block  cache block if not under UMM control, null otherwise
	 * @param dirty  indicator if block is dirty (subject to buffer pool management)
	 * @return       pinned cache block, potentially restored from local FS
	 */
	public CacheBlock pin(String key, CacheBlock block, boolean dirty) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Pins a virtual cache block into operation memory, by making a size reservation.
	 * The provided size is an upper bound of the actual object size, and can be
	 * updated on unpin (once the actual cache block is provided).
	 * 
	 * @param key    unique identifier and local FS filename for eviction
	 * @param size   memory reservation in operation area
	 * @param dirty  indicator if block is dirty (subject to buffer pool management)
	 */
	public void pin(String key, long size, boolean dirty) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Unpins (releases) a cache block from operation memory. Dirty objects
	 * are logically moved back to the buffer pool area.
	 * 
	 * @param key    unique identifier and local FS filename for eviction
	 */
	public void unpin(String key) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Unpins (releases) a cache block from operation memory. If the size of
	 * the provided cache block differs from the UMM meta data, the UMM meta
	 * data is updated. Use cases include update-in-place operations and
	 * size reservations via worst-case upper bound estimates.
	 * 
	 * @param key    unique identifier and local FS filename for eviction
	 * @param block  cache block which may be under UMM control, if null ignored
	 */
	public void unpin(String key, CacheBlock block) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Removes a cache block associated with the given key from all memory
	 * areas, and deletes evicted representations (files in local FS). The
	 * local file system deletes can happen asynchronously.
	 * 
	 * @param key    unique identifier and local FS filename for eviction
	 */
	public void delete(String key) {
		//TODO implement
		throw new NotImplementedException();
	}
	
	/**
	 * Removes all cache blocks from all memory areas and deletes all evicted
	 * representations (files in local FS). All internally thread pools must be
	 * shut down in a gracefully manner (e.g., wait for pending deletes).
	 */
	public void deleteAll() {
		//TODO implement
		throw new NotImplementedException();
	}
}
