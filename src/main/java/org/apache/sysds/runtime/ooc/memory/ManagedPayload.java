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

package org.apache.sysds.runtime.ooc.memory;

import java.util.concurrent.atomic.AtomicBoolean;

public final class ManagedPayload<T> {
	private final T _value;
	private final long _bytes;
	private final MemoryAllowance _owner;
	private final AtomicBoolean _transferred;

	public ManagedPayload(T value, long bytes, MemoryAllowance owner) {
		if(value == null)
			throw new IllegalArgumentException("Managed payload requires a value.");
		if(bytes < 0)
			throw new IllegalArgumentException("Managed payload bytes must not be negative: " + bytes);
		if(bytes > 0 && owner == null)
			throw new IllegalArgumentException("Managed payload with charged bytes requires an owning allowance.");
		_value = value;
		_bytes = bytes;
		_owner = owner;
		_transferred = new AtomicBoolean(false);
	}

	public T value() {
		return _value;
	}

	public long bytes() {
		return _bytes;
	}

	public MemoryAllowance owner() {
		return _owner;
	}

	public void transfer() {
		if(!_transferred.compareAndSet(false, true))
			throw new IllegalStateException("Managed payload was already settled.");
	}

	public void release() {
		if(_transferred.compareAndSet(false, true) && _bytes > 0)
			_owner.release(_bytes);
	}

	public boolean isTransferred() {
		return _transferred.get();
	}
}
