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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.ArrayDeque;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

public final class OrderedMaterializedStoreReader<T extends SpillableObject> implements MaterializedStore.StoreReader {
	private static final Request CLOSED = new Request(-1, OOCFuture.completed(null));

	private final OOCCache _cache;
	private final long _streamId;
	private final MaterializedStore.AccessPattern _pattern;
	private final MemoryAllowance _allowance;
	private final int _maxPrefetch;
	private final boolean _softOrdering;
	private final Runnable _afterClose;
	private final IntConsumer _afterRelease;
	private final BlockingQueue<Request> _requests;
	private final AtomicInteger _inFlightRequests;
	private volatile boolean _closed;

	OrderedMaterializedStoreReader(OOCCache cache, long streamId, MaterializedStore.AccessPattern pattern,
		MemoryAllowance allowance, int maxPrefetch, boolean softOrdering, Runnable afterClose,
		IntConsumer afterRelease) {
		_cache = cache;
		_streamId = streamId;
		_pattern = pattern;
		_allowance = allowance;
		_maxPrefetch = maxPrefetch;
		_softOrdering = softOrdering;
		_afterClose = afterClose;
		_afterRelease = afterRelease;
		_requests = new LinkedBlockingQueue<>();
		_inFlightRequests = new AtomicInteger();
	}

	@Override
	public MaterializedStore.Liveness liveness() {
		return _pattern;
	}

	@Override
	public boolean isClosed() {
		return _closed;
	}

	@Override
	public void close() {
		ArrayDeque<Request> pending = new ArrayDeque<>();
		synchronized(this) {
			if(_closed)
				return;
			_closed = true;
			_requests.drainTo(pending);
		}
		for(Request request : pending) {
			releaseWhenReady(request);
			if(_softOrdering)
				_inFlightRequests.decrementAndGet();
		}
		if(_softOrdering)
			_requests.offer(CLOSED);
		_afterClose.run();
	}

	public boolean hasNext() {
		if(_softOrdering) {
			checkReady();
			fillSoft();
			return _inFlightRequests.get() > 0;
		}
		synchronized(this) {
			checkReady();
			fillStrict();
			return !_requests.isEmpty();
		}
	}

	public StoreLease<T> next() throws InterruptedException {
		if(_softOrdering) {
			checkReady();
			return nextSoft();
		}
		Request request;
		synchronized(this) {
			checkReady();
			fillStrict();
			if(_requests.isEmpty())
				throw new IllegalStateException("No remaining item");
			request = _requests.remove();
		}
		BlockEntry entry;
		try {
			entry = awaitEntry(request);
		}
		catch(InterruptedException | RuntimeException ex) {
			releaseWhenReady(request);
			throw ex;
		}
		synchronized(this) {
			if(_closed) {
				_cache.unpin(entry, _allowance);
				throw new IllegalStateException("Reader is closed");
			}
			try {
				fillStrict();
			}
			catch(RuntimeException ex) {
				_cache.unpin(entry, _allowance);
				throw ex;
			}
		}
		return new StoreLease<>(entry, () -> release(request._index, entry));
	}

	public void release(int index, BlockEntry entry) {
		_cache.unpin(entry, _allowance);
		_pattern.consumed(index);
		_afterRelease.accept(index);
	}

	private void checkReady() {
		if(_closed)
			throw new IllegalStateException("Reader is closed");
	}

	private StoreLease<T> nextSoft() throws InterruptedException {
		fillSoft();
		if(_inFlightRequests.get() <= 0)
			throw new IllegalStateException("No remaining item");
		Request request = _requests.take();
		if(request == CLOSED)
			throw new IllegalStateException("Reader is closed");
		_inFlightRequests.decrementAndGet();
		BlockEntry entry;
		try {
			entry = request._future.get();
		}
		catch(ExecutionException e) {
			throw DMLRuntimeException.of(e.getCause());
		}
		if(entry == null)
			throw new IllegalStateException("Reader is closed");
		fillSoft();
		return new StoreLease<>(entry, () -> release(request._index, entry));
	}

	private void fillStrict() {
		while(_requests.size() < _maxPrefetch && _pattern.hasNext()) {
			int index = _pattern.next();
			OOCFuture<BlockEntry> future = _cache.pin(_streamId, index, _allowance);
			_requests.offer(new Request(index, future));
		}
	}

	private void fillSoft() {
		while(_inFlightRequests.get() < _maxPrefetch && _pattern.hasNext()) {
			int index = _pattern.next();
			OOCFuture<BlockEntry> future = _cache.pin(_streamId, index, _allowance);
			_inFlightRequests.incrementAndGet();
			registerSoftRequest(new Request(index, future));
		}
	}

	private void registerSoftRequest(Request request) {
		request._future.whenComplete((entry, error) -> {
			if(error != null || entry != null) {
				completeSoft(request);
				return;
			}
			request._future = OOCUtils.pinAdmitted(_cache, _streamId, request._index, _allowance, () -> _closed);
			request._future.whenComplete((admittedEntry, admittedError) -> completeSoft(request));
		});
	}

	private void completeSoft(Request request) {
		if(_closed) {
			releaseWhenReady(request);
			_inFlightRequests.decrementAndGet();
			return;
		}
		_requests.offer(request);
		if(_closed && _requests.remove(request)) {
			releaseWhenReady(request);
			_inFlightRequests.decrementAndGet();
		}
	}

	private BlockEntry awaitEntry(Request request) throws InterruptedException {
		try {
			BlockEntry entry = request._future.get();
			if(entry == null) {
				OOCFuture<BlockEntry> retried = OOCUtils.pinAdmitted(_cache, _streamId, request._index, _allowance,
					() -> _closed);
				request._future = retried;
				entry = retried.get();
				if(entry == null)
					throw new IllegalStateException("Reader is closed");
			}
			return entry;
		}
		catch(ExecutionException e) {
			throw DMLRuntimeException.of(e.getCause());
		}
	}

	private void releaseWhenReady(Request request) {
		request._future.whenComplete((entry, error) -> {
			if(entry != null)
				_cache.unpin(entry, _allowance);
		});
	}

	private static final class Request {
		private final int _index;
		private OOCFuture<BlockEntry> _future;

		private Request(int index, OOCFuture<BlockEntry> future) {
			_index = index;
			_future = future;
		}
	}
}
