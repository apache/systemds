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

package org.apache.sysds.runtime.ooc.store;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.function.IntConsumer;
import java.util.function.IntSupplier;

public final class IndexedMaterializedStoreReader<T extends SpillableObject> implements MaterializedStore.StoreReader {
	private final OOCCache _cache;
	private final long _streamId;
	private final IntSupplier _completedSize;
	private final MaterializedStore.Liveness _liveness;
	private final Runnable _afterClose;
	private final IntConsumer _afterRelease;
	private volatile boolean _closed;

	IndexedMaterializedStoreReader(OOCCache cache, long streamId, IntSupplier completedSize,
		MaterializedStore.Liveness liveness, Runnable afterClose, IntConsumer afterRelease) {
		_cache = cache;
		_streamId = streamId;
		_completedSize = completedSize;
		_liveness = liveness;
		_afterClose = afterClose;
		_afterRelease = afterRelease;
	}

	@Override
	public MaterializedStore.Liveness liveness() {
		return _liveness;
	}

	@Override
	public boolean isClosed() {
		return _closed;
	}

	@Override
	public void close() {
		if(_closed)
			return;
		_closed = true;
		_afterClose.run();
	}

	public OOCFuture<StoreLease<T>> request(int index, MemoryAllowance requestAllowance) {
		checkReady(index);
		reserve(index);
		OOCFuture<BlockEntry> pinned = OOCUtils.pinAdmitted(_cache, _streamId, index, requestAllowance, () -> _closed);
		OOCFuture<StoreLease<T>> result = new OOCFuture<>();
		pinned.whenComplete((entry, error) -> {
			if(error != null) {
				_liveness.unreserve(index);
				result.completeExceptionally(error);
			}
			else if(entry == null) {
				_liveness.unreserve(index);
				result.complete(null);
			}
			else
				result.complete(new StoreLease<>(entry, () -> release(index, entry, requestAllowance)));
		});
		return result;
	}

	public StoreLease<T> requestIfLive(int index, MemoryAllowance requestAllowance) {
		checkReady(index);
		reserve(index);
		BlockEntry entry = _cache.pinIfLive(_streamId, index, requestAllowance);
		if(entry == null) {
			_liveness.unreserve(index);
			return null;
		}
		return new StoreLease<>(entry, () -> release(index, entry, requestAllowance));
	}

	private void release(int index, BlockEntry entry, MemoryAllowance requestAllowance) {
		_cache.unpin(entry, requestAllowance);
		_liveness.consumed(index);
		_afterRelease.accept(index);
	}

	private void reserve(int index) {
		if(!_liveness.reserve(index))
			throw new IllegalStateException("Index is no longer live for this reader: " + index);
	}

	private void checkReady(int index) {
		if(_closed)
			throw new IllegalStateException("Reader is closed");
		if(index < 0 || index >= _completedSize.getAsInt())
			throw new IndexOutOfBoundsException("Invalid requested index: " + index);
	}
}
