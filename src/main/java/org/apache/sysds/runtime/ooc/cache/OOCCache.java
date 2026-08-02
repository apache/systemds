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

import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;

import java.util.function.LongUnaryOperator;

public interface OOCCache {
	/**
	 * Pins an item backed by an allowance. A successful pin transfers memory ownership from the cache to the owner of
	 * the allowance and guarantees data availability. While pinned, the bytes of the entry are not counted as
	 * cache-owned memory.
	 *
	 * @param key
	 * @param allowance
	 * @return a non-null future of the pinned block entry; the future result is null if the required memory could not
	 *         be reserved
	 */
	default OOCFuture<BlockEntry> pin(BlockKey key, MemoryAllowance allowance) {
		return pin(key.getStreamId(), key.getSequenceNumber(), allowance);
	}

	/**
	 * Pins an item backed by an allowance. If the allowance cannot reserve enough memory, this method will wait until
	 * memory is available. A successful pin transfers memory ownership from the cache to the owner of the allowance and
	 * guarantees data availability. While pinned, the bytes of the entry are not counted as cache-owned memory.
	 *
	 * @param key
	 * @param allowance
	 * @return a non-null future of the pinned block entry
	 */
	default OOCFuture<BlockEntry> pinAdmitted(BlockKey key, MemoryAllowance allowance) {
		return pinAdmitted(key.getStreamId(), key.getSequenceNumber(), allowance);
	}

	/**
	 * Adds a new pinned entry whose bytes are already owned by the given allowance. Ownership can later move only via
	 * pin/unpin.
	 */
	default BlockEntry putPinned(BlockKey key, Object data, long size, MemoryAllowance allowance) {
		return putPinned(key.getStreamId(), key.getSequenceNumber(), data, size, allowance);
	}

	/**
	 * Adds a new pinned entry whose bytes are already owned by the given allowance. Ownership can later move only via
	 * pin/unpin.
	 */
	BlockEntry putPinned(long sId, long tId, Object data, long size, MemoryAllowance allowance);

	/**
	 * Pins an item backed by an allowance. A successful pin transfers memory ownership from the cache to the owner of
	 * the allowance and guarantees data availability. While pinned, the bytes of the entry are not counted as
	 * cache-owned memory.
	 *
	 * @param sId
	 * @param tId
	 * @param allowance
	 * @return a non-null future of the pinned block entry; the future result is null if the required memory could not
	 *         be reserved
	 */
	OOCFuture<BlockEntry> pin(long sId, long tId, MemoryAllowance allowance);

	/**
	 * Pins an item backed by an allowance. If the allowance cannot reserve enough memory, this method will wait until
	 * memory is available. A successful pin transfers memory ownership from the cache to the owner of the allowance and
	 * guarantees data availability. While pinned, the bytes of the entry are not counted as cache-owned memory.
	 *
	 * @param sId
	 * @param tId
	 * @param allowance
	 * @return a non-null future of the pinned block entry
	 */
	OOCFuture<BlockEntry> pinAdmitted(long sId, long tId, MemoryAllowance allowance);

	/**
	 * Pins an item backed by an allowance if it is already live in cache. A successful pin transfers memory ownership
	 * from the cache to the owner of the allowance and guarantees data availability. While pinned, the bytes of the
	 * entry are not counted as cache-owned memory. Implementations must reserve the required bytes from the allowance
	 * before making data available.
	 *
	 * @param sId
	 * @param tId
	 * @param allowance
	 * @return the pinned block entry if available. Null if the required memory could not be reserved or the block is
	 *         not live
	 */
	BlockEntry pinIfLive(long sId, long tId, MemoryAllowance allowance);

	/**
	 * Unpins an item that is still backed by the given allowance. Unpinning tries to transfer memory ownership back to
	 * the cache. An ownership transfer may commit immediately only if this does not cause the cache to exceed its hard
	 * limit. Otherwise, the transfer is deferred and the allowance remains charged until the returned handle commits,
	 * is reclaimed, or is superseded by a later pin that transfers ownership to another allowance. Unpin can be viewed
	 * as an eventually resolving operation.
	 *
	 * @param entry
	 * @param allowance
	 * @return a handle describing the ownership transfer from allowance-owned memory back to cache-owned memory
	 */
	UnpinHandle unpin(BlockEntry entry, MemoryAllowance allowance);

	/**
	 * Referencing a pinned entry guarantees that its key remains in the cache until dereferenced.
	 *
	 * @param entry
	 * @return
	 */
	int reference(BlockEntry entry);

	/**
	 * Dereferencing allows an entry to be forgotten if no further reference is held. Dereferencing may not immediately
	 * cause entry removal if still pinned.
	 *
	 * @param entry
	 * @return
	 */
	int dereference(BlockEntry entry);

	/**
	 * Dereferencing allows an entry to be forgotten if no further reference is held. Dereferencing may not immediately
	 * cause entry removal if still pinned.
	 */
	int dereference(BlockKey key);

	void updateLimits(long hardLimit, long evictionLimit);

	/**
	 * Adds an eviction scoring policy for one logical cache stream. Larger scores are selected for eviction first.
	 * {@link Long#MAX_VALUE} remains reserved as "no policy score".
	 */
	void addEvictionPolicy(long streamId, LongUnaryOperator scoreFn);

	/**
	 * Returns the current cache-owned size in bytes.
	 */
	long getOwnedCacheSize();

	void shutdown();

	interface UnpinHandle {
		BlockEntry entry();

		MemoryAllowance allowance();

		long bytes();

		boolean isCommitted();

		OOCFuture<Boolean> getCompletionFuture();
	}
}
