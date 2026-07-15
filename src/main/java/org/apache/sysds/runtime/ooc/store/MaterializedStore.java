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
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public final class MaterializedStore<T extends SpillableObject> {
	private final OOCCache _cache;
	private final long _streamId;
	private final ArrayList<StoreReader> _registeredReaders;
	private final BitSet _forgotten;
	private final AtomicInteger _published;
	private final AtomicInteger _publishedCount;

	private volatile List<StoreReader> _readers;
	private volatile int _completedSize;
	private volatile boolean _complete;
	private volatile boolean _readersSealed;
	private volatile boolean _closed;

	public MaterializedStore(OOCCache cache, long streamId) {
		_cache = cache;
		_streamId = streamId;
		_registeredReaders = new ArrayList<>();
		_forgotten = new BitSet();
		_published = new AtomicInteger();
		_publishedCount = new AtomicInteger();
		_readers = Collections.emptyList();
	}

	StoreLease<T> publishPinnedLive(int index, T value, long bytes, MemoryAllowance allowance) {
		BlockEntry entry;
		try {
			if(_complete || _closed)
				throw new IllegalStateException("Store no longer accepts published items");
			if(index < 0 || index == Integer.MAX_VALUE)
				throw new IndexOutOfBoundsException("Invalid index: " + index);
			entry = _cache.putPinned(_streamId, index, value, bytes, allowance);
		}
		catch(RuntimeException ex) {
			if(bytes > 0)
				allowance.release(bytes);
			throw ex;
		}
		_publishedCount.incrementAndGet();
		updatePublished(index + 1);
		return new StoreLease<>(entry, () -> {
			_cache.unpin(entry, allowance);
			tryForget(index);
		});
	}

	StoreLease<T> publishPinnedLive(int index, ManagedPayload<T> payload) {
		payload.transfer();
		return publishPinnedLive(index, payload.value(), payload.bytes(), payload.owner());
	}

	public synchronized void complete() {
		if(_complete)
			return;
		_completedSize = _published.get();
		if(_publishedCount.get() != _completedSize)
			throw new IllegalStateException("Incomplete publication: " + _publishedCount.get()
				+ " published items for logical range [0, " + _completedSize + ")");
		_complete = true;
	}

	public synchronized OrderedMaterializedStoreReader<T> openReader(AccessPattern pattern, MemoryAllowance allowance,
		int maxPrefetch) {
		return openReader(pattern, allowance, maxPrefetch, true);
	}

	public synchronized OrderedMaterializedStoreReader<T> openReader(AccessPattern pattern, MemoryAllowance allowance,
		int maxPrefetch, boolean softOrdering) {
		if(!_complete || _closed)
			throw new IllegalStateException("Readers require a completed store");
		if(_readersSealed)
			throw new IllegalStateException("Store no longer accepts new readers");
		OrderedMaterializedStoreReader<T> reader = new OrderedMaterializedStoreReader<>(_cache, _streamId, pattern,
			allowance, Math.max(1, maxPrefetch), softOrdering, this::forgetAfterReaderClose, this::tryForget);
		_registeredReaders.add(reader);
		return reader;
	}

	public synchronized IndexedMaterializedStoreReader<T> openIndexedReader(Liveness liveness) {
		if(!_complete || _closed)
			throw new IllegalStateException("Readers require a completed store");
		if(_readersSealed)
			throw new IllegalStateException("Store no longer accepts new readers");
		IndexedMaterializedStoreReader<T> reader = new IndexedMaterializedStoreReader<>(_cache, _streamId,
			() -> _completedSize, liveness, this::forgetAfterReaderClose, this::tryForget);
		_registeredReaders.add(reader);
		return reader;
	}

	public OOCFuture<StoreLease<T>> requestPublished(int index, MemoryAllowance allowance) {
		if(_closed)
			throw new IllegalStateException("Store is closed");
		if(index < 0 || index >= _published.get())
			throw new IndexOutOfBoundsException("Invalid requested index: " + index);
		OOCFuture<BlockEntry> pinned = OOCUtils.pinAdmitted(_cache, _streamId, index, allowance, () -> _closed);
		OOCFuture<StoreLease<T>> result = new OOCFuture<>();
		pinned.whenComplete((entry, error) -> {
			if(error != null)
				result.completeExceptionally(error);
			else if(entry == null)
				result.complete(null);
			else
				result.complete(new StoreLease<>(entry, () -> _cache.unpin(entry, allowance)));
		});
		return result;
	}

	public synchronized void sealReaders() {
		if(_closed)
			throw new IllegalStateException("Cannot seal readers for a closed store");
		if(_readersSealed)
			return;
		_readers = new ArrayList<>(_registeredReaders);
		_readersSealed = true;
		int publishedSize = _complete ? _completedSize : _published.get();
		for(int i = 0; i < publishedSize; i++)
			tryForget(i);
	}

	public int size() {
		return _complete ? _completedSize : _published.get();
	}

	public void close() {
		List<StoreReader> localReaders;
		synchronized(this) {
			if(_closed)
				return;
			_closed = true;
			localReaders = _readersSealed ? _readers : new ArrayList<>(_registeredReaders);
		}
		for(StoreReader localReader : localReaders)
			localReader.close();
		for(int i = 0; i < size(); i++)
			if(markForgotten(i))
				_cache.dereference(new BlockKey(_streamId, i));
	}

	private void tryForget(int index) {
		if(!_readersSealed)
			return;
		List<StoreReader> localReaders = _readers;
		for(StoreReader reader : localReaders)
			if(!reader.isClosed() && reader.liveness().needs(index))
				return;
		if(markForgotten(index))
			_cache.dereference(new BlockKey(_streamId, index));
	}

	private synchronized boolean markForgotten(int index) {
		if(_forgotten.get(index))
			return false;
		_forgotten.set(index);
		return true;
	}

	private void forgetAfterReaderClose() {
		if(_closed || !_readersSealed)
			return;
		for(int i = 0; i < _completedSize; i++)
			tryForget(i);
	}

	private void updatePublished(int size) {
		int current = _published.get();
		while(current < size && !_published.compareAndSet(current, size))
			current = _published.get();
	}

	public interface Liveness {
		boolean needs(int index);

		void consumed(int index);

		default boolean reserve(int index) {
			return needs(index);
		}

		default void unreserve(int index) {
		}
	}

	public interface AccessPattern extends Liveness {
		boolean hasNext();

		int next();
	}

	public interface StoreReader extends AutoCloseable {
		Liveness liveness();

		boolean isClosed();

		@Override
		void close();
	}
}
