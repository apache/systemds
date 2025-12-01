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

package org.apache.sysds.runtime.ooc.cache;

import java.util.List;
import java.util.concurrent.CompletableFuture;

public interface OOCCacheScheduler {

	/**
	 * Requests a single block from the cache.
	 * @param key the requested key associated to the block
	 * @return the available BlockEntry
	 */
	CompletableFuture<BlockEntry> request(BlockKey key);

	/**
	 * Requests a list of blocks from the cache that must be available at the same time.
	 * @param keys the requested keys associated to the block
	 * @return the list of available BlockEntries
	 */
	CompletableFuture<List<BlockEntry>> request(List<BlockKey> keys);

	/**
	 * Places a new block in the cache. Note that objects are immutable and cannot be overwritten.
	 * The object data should now only be accessed via cache, as ownership has been transferred.
	 * @param key the associated key of the block
	 * @param data the block data
	 * @param size the size of the data
	 */
	void put(BlockKey key, Object data, long size);

	/**
	 * Places a new block in the cache and returns a pinned handle.
	 * Note that objects are immutable and cannot be overwritten.
	 * @param key the associated key of the block
	 * @param data the block data
	 * @param size the size of the data
	 */
	BlockEntry putAndPin(BlockKey key, Object data, long size);

	/**
	 * Forgets a block from the cache.
	 * @param key the associated key of the block
	 */
	void forget(BlockKey key);

	/**
	 * Pins a BlockEntry in cache to prevent eviction.
	 * @param entry the entry to be pinned
	 */
	void pin(BlockEntry entry);

	/**
	 * Unpins a pinned block.
	 * @param entry the entry to be unpinned
	 */
	void unpin(BlockEntry entry);

	/**
	 * Shuts down the cache scheduler.
	 */
	void shutdown();
}
