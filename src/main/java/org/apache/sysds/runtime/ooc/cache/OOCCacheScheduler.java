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
	 * Tries to request a list of blocks from the cache that must be available at the same time.
	 * Immediately returns the list of entries if present, otherwise null without scheduling reads.
	 * @param keys the requested keys associated to the block
	 * @return the list of available BlockEntries
	 */
	List<BlockEntry> tryRequest(List<BlockKey> keys);

	/**
	 * Requests any n entries of the list of blocks, preferring an available item.
	 */
	CompletableFuture<List<BlockEntry>> requestAnyOf(List<BlockKey> keys, int n, List<BlockKey> selectionOut);

	/**
	 * Requests any n entries of the list of blocks, preferring an available item.
	 */
	List<BlockEntry> tryRequestAnyOf(List<BlockKey> keys, int n, List<BlockKey> selectionOut);

	/**
	 * Adds the given priority to any pending request accessing the key.
	 * Multi-requests are prioritized partially.
	 */
	void prioritize(BlockKey key, double priority);

	/**
	 * Places a new block in the cache. Note that objects are immutable and cannot be overwritten.
	 * The object data should now only be accessed via cache, as ownership has been transferred.
	 * @param key the associated key of the block
	 * @param data the block data
	 * @param size the size of the data
	 */
	BlockKey put(BlockKey key, Object data, long size);

	/**
	 * Places a new block in the cache and returns a pinned handle.
	 * Note that objects are immutable and cannot be overwritten.
	 * @param key the associated key of the block
	 * @param data the block data
	 * @param size the size of the data
	 */
	BlockEntry putAndPin(BlockKey key, Object data, long size);

	/**
	 * Places a new source-backed block in the cache and registers the location with the IO handler. The entry is
	 * treated as backed by disk, so eviction does not schedule spill writes.
	 *
	 * @param key        the associated key of the block
	 * @param data       the block data
	 * @param size       the size of the data
	 * @param descriptor the source location descriptor
	 */
	void putSourceBacked(BlockKey key, Object data, long size, OOCIOHandler.SourceBlockDescriptor descriptor);

	/**
	 * Places a new source-backed block in the cache and returns a pinned handle.
	 *
	 * @param key        the associated key of the block
	 * @param data       the block data
	 * @param size       the size of the data
	 * @param descriptor the source location descriptor
	 */
	BlockEntry putAndPinSourceBacked(BlockKey key, Object data, long size,
		OOCIOHandler.SourceBlockDescriptor descriptor);

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
	 * Returns the current cache size in bytes.
	 */
	long getCacheSize();

	/**
	 * Returns if the current cache size is within its defined memory limits.
	 */
	boolean isWithinLimits();

	/**
	 * Returns if the current cache size is within its soft memory limits.
	 */
	boolean isWithinSoftLimits();

	/**
	 * Shuts down the cache scheduler.
	 */
	void shutdown();

	/**
	 * Updates the cache limits.
	 */
	void updateLimits(long evictionLimit, long hardLimit);
}
