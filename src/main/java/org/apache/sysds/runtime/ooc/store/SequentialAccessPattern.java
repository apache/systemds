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

import org.apache.sysds.runtime.ooc.cache.collections.ConcurrentBitSet;

public final class SequentialAccessPattern implements MaterializedStore.AccessPattern {
	private final int _size;
	private final ConcurrentBitSet _consumed;
	private int _next;
	private volatile int _consumedThrough;

	public SequentialAccessPattern(int size) {
		if(size < 0)
			throw new IllegalArgumentException("Size must not be negative: " + size);
		_size = size;
		_consumed = new ConcurrentBitSet(Math.max(1, size));
		_next = 0;
		_consumedThrough = -1;
	}

	@Override
	public boolean hasNext() {
		return _next < _size;
	}

	@Override
	public int next() {
		if(!hasNext())
			throw new IllegalStateException("No remaining index");
		return _next++;
	}

	@Override
	public boolean needs(int index) {
		return index >= 0 && index < _size && index > _consumedThrough && !_consumed.get(index);
	}

	@Override
	public synchronized void consumed(int index) {
		if(index < 0 || index >= _size)
			throw new IndexOutOfBoundsException("Invalid consumed index: " + index);
		_consumed.set(index);
		while(_consumedThrough + 1 < _size && _consumed.get(_consumedThrough + 1))
			_consumedThrough++;
	}
}
