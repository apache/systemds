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
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;

import java.util.concurrent.atomic.AtomicInteger;

public final class StoreLease<T extends SpillableObject> implements AutoCloseable {
	private final Runnable _releaser;
	private final T _value;
	private final BlockEntry _entry;
	private final AtomicInteger _shared;
	private boolean _open;

	StoreLease(BlockEntry entry, Runnable releaser) {
		this(null, entry, releaser, new AtomicInteger(1));
	}

	StoreLease(T value, Runnable releaser) {
		this(value, null, releaser, new AtomicInteger(1));
	}

	private StoreLease(T value, BlockEntry entry, Runnable releaser, AtomicInteger shared) {
		_releaser = releaser;
		_value = value;
		_entry = entry;
		_shared = shared;
		_open = true;
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
		_shared.incrementAndGet();
		return new StoreLease<>(_value, _entry, _releaser, _shared);
	}

	@Override
	public synchronized void close() {
		if(!_open)
			return;
		_open = false;
		if(_shared.decrementAndGet() == 0)
			_releaser.run();
	}
}
