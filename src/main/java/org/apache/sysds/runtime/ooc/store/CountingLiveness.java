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

import java.util.concurrent.atomic.AtomicIntegerArray;

public final class CountingLiveness implements MaterializedStore.Liveness {
	private final AtomicIntegerArray _remaining;
	private final AtomicIntegerArray _reservable;

	public CountingLiveness(int size, int count) {
		if(size < 0 || count < 0)
			throw new IllegalArgumentException("Invalid args: size=" + size + ", count=" + count);
		_remaining = new AtomicIntegerArray(size);
		_reservable = new AtomicIntegerArray(size);
		for(int i = 0; i < size; i++) {
			_remaining.set(i, count);
			_reservable.set(i, count);
		}
	}

	@Override
	public boolean needs(int index) {
		return index >= 0 && index < _remaining.length() && _remaining.get(index) > 0;
	}

	@Override
	public void consumed(int index) {
		decrement(_remaining, index);
	}

	@Override
	public boolean reserve(int index) {
		return index >= 0 && index < _reservable.length() && decrement(_reservable, index);
	}

	@Override
	public void unreserve(int index) {
		_reservable.incrementAndGet(index);
	}

	private static boolean decrement(AtomicIntegerArray counters, int index) {
		while(true) {
			int current = counters.get(index);
			if(current <= 0)
				return false;
			if(counters.compareAndSet(index, current, current - 1))
				return true;
		}
	}
}
