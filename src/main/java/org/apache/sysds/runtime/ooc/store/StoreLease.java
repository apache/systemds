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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;

public final class StoreLease<T extends SpillableObject> implements AutoCloseable {
	private final T _value;
	private final BlockEntry _entry;
	private final SharedStoreLease _sharedLease;
	private boolean _open;

	private StoreLease(T value, BlockEntry entry, SharedStoreLease release) {
		_value = value;
		_entry = entry;
		_sharedLease = release;
		_open = true;
	}

	public static <T extends SpillableObject> StoreLease<T> create(T value, Runnable releaser) {
		return new StoreLease<>(value, null, new SharedStoreLease(() -> {
			releaser.run();
			return OOCFuture.completed(null);
		}, new AtomicInteger(1), new OOCFuture<>()));
	}

	public static <T extends SpillableObject> StoreLease<T> create(BlockEntry entry, Runnable releaser) {
		return new StoreLease<>(null, entry, new SharedStoreLease(() -> {
			releaser.run();
			return OOCFuture.completed(null);
		}, new AtomicInteger(1), new OOCFuture<>()));
	}

	public static <T extends SpillableObject> StoreLease<T> createAsync(BlockEntry entry,
		Supplier<? extends OOCFuture<?>> releaser) {
		return new StoreLease<>(null, entry, new SharedStoreLease(releaser, new AtomicInteger(1), new OOCFuture<>()));
	}

	@SuppressWarnings("unchecked")
	public synchronized T value() {
		if(!_open)
			throw new IllegalStateException("Lease is closed");
		return _entry == null ? _value : (T) _entry.getData();
	}

	synchronized BlockEntry entry() {
		if(!_open)
			throw new IllegalStateException("Lease is closed");
		return _entry;
	}

	public synchronized StoreLease<T> retain() {
		if(!_open)
			throw new IllegalStateException("Lease is closed");
		_sharedLease.references.incrementAndGet();
		return new StoreLease<>(_value, _entry, _sharedLease);
	}

	public OOCFuture<Void> closeAsync() {
		boolean release;
		synchronized(this) {
			if(!_open)
				return _sharedLease.future;
			_open = false;
			release = _sharedLease.references.decrementAndGet() == 0;
		}
		if(release) {
			OOCFuture<?> released;
			try {
				released = _sharedLease.releaser.get();
			}
			catch(Throwable error) {
				_sharedLease.future.completeExceptionally(error);
				return _sharedLease.future;
			}
			if(released == null) {
				_sharedLease.future
					.completeExceptionally(new NullPointerException("Asynchronous lease releaser returned null"));
				return _sharedLease.future;
			}
			released.whenComplete((ignored, error) -> {
				if(error == null)
					_sharedLease.future.complete(null);
				else
					_sharedLease.future.completeExceptionally(error);
			});
		}
		return _sharedLease.future;
	}

	@Override
	public void close() {
		closeAsync();
	}

	private record SharedStoreLease(Supplier<? extends OOCFuture<?>> releaser, AtomicInteger references,
		OOCFuture<Void> future) {
	}
}
