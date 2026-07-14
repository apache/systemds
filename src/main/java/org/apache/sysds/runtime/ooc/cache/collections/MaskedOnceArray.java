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

package org.apache.sysds.runtime.ooc.cache.collections;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.function.Consumer;

public class MaskedOnceArray<T> {
	private static final int RETIRED = Integer.MIN_VALUE;
	private static final VarHandle NON_NULL_COUNT;

	static {
		try {
			NON_NULL_COUNT = MethodHandles.lookup().findVarHandle(MaskedOnceArray.class, "_nonNullCount", int.class);
		}
		catch(ReflectiveOperationException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private final AtomicReferenceArray<T> _values;
	protected final ConcurrentBitSet _liveState;
	private volatile int _nonNullCount;

	public MaskedOnceArray(int length) {
		_values = new AtomicReferenceArray<>(length);
		_liveState = new ConcurrentBitSet(length);
		_nonNullCount = 0;
	}

	public boolean put(int i, T value) {
		if(value == null) {
			return clear(i);
		}
		if(!incrementNonNullCount())
			return false;
		boolean changed = _values.getAndSet(i, value) == null;
		if(!changed)
			decrementNonNullCount();
		_liveState.set(i);
		return changed;
	}

	private boolean incrementNonNullCount() {
		while(true) {
			int count = (int) NON_NULL_COUNT.getAcquire(this);
			if(count == RETIRED)
				return false;
			if(NON_NULL_COUNT.compareAndSet(this, count, count + 1))
				return true;
		}
	}

	private void decrementNonNullCount() {
		while(true) {
			int count = (int) NON_NULL_COUNT.getAcquire(this);
			if(count <= 0)
				return;
			if(NON_NULL_COUNT.compareAndSet(this, count, count - 1))
				return;
		}
	}

	public boolean clear(int i) {
		boolean changed = _values.getAndSet(i, null) != null;
		if(changed)
			decrementNonNullCount();
		_liveState.clear(i);
		return changed;
	}

	public T get(int i) {
		return _values.get(i);
	}

	public void forEachVisible(Consumer<? super T> action) {
		for(int i = 0; i < _values.length(); i++) {
			T v = _values.get(i);
			if(v != null)
				action.accept(v);
		}
	}

	public boolean tryRetireIfEmpty() {
		return NON_NULL_COUNT.compareAndSet(this, 0, RETIRED);
	}

	public boolean isRetired() {
		return (int) NON_NULL_COUNT.getAcquire(this) == RETIRED;
	}

	public boolean isEmpty() {
		return (int) NON_NULL_COUNT.getAcquire(this) == 0;
	}

	public void setLive(int i) {
		_liveState.set(i);
	}

	public void clearLive(int i) {
		_liveState.clear(i);
	}

	public boolean forEachLive(IndexedObjectPredicate<? super T> action, boolean reversed, int offset) {
		if(reversed)
			return forEachLiveBackward(action, offset);
		else
			return forEachLiveForward(action, offset);
	}

	private boolean forEachLiveForward(IndexedObjectPredicate<? super T> action, int offset) {
		int len = _liveState.length();
		T data;
		for(int word = 0; word < len; word++) {
			if(_liveState.getWord(word) == 0)
				continue;
			int lower = word * 64;
			int upper = (word + 1) * 64;
			for(int i = lower; i < upper; i++) {
				data = get(i);
				if(data != null)
					if(!action.test(offset + i, data))
						return false;
			}
		}
		return true;
	}

	private boolean forEachLiveBackward(IndexedObjectPredicate<? super T> action, int offset) {
		int len = _liveState.length();
		for(int word = len - 1; word >= 0; word--) {
			if(_liveState.getWord(word) == 0)
				continue;
			int lower = word * 64;
			int upper = (word + 1) * 64;
			T data;
			for(int i = upper - 1; i >= lower; i--) {
				data = get(i);
				if(data != null)
					if(!action.test(offset + i, data))
						return false;
			}
		}
		return true;
	}
}
