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

public final class BlockEntry {
	private final BlockKey _key;
	private final long _size;
	private volatile int _pinCount;
	private volatile BlockState _state;
	private Object _data;
	private int _retainHintCount;
	private int _referenceCount; // The number of references from different managing instances (e.g. CachingStream)
	// Optional implementation-local cache metadata; null for cache implementations that do not need it.
	private volatile Object _cacheMeta;

	public BlockEntry(BlockKey key) {
		this._key = key;
		this._size = -1;
		this._pinCount = 0;
		this._state = BlockState.COLD;
		this._data = null;
		this._retainHintCount = 0;
		this._referenceCount = 0;
		this._cacheMeta = null;
	}

	public BlockEntry(BlockKey key, long size, Object data) {
		this._key = key;
		this._size = size;
		this._pinCount = 0;
		this._state = BlockState.HOT;
		this._data = data;
		this._retainHintCount = 0;
		this._referenceCount = 1;
		this._cacheMeta = null;
	}

	public BlockEntry(BlockKey key, long size, Object data, BlockState state) {
		this._key = key;
		this._size = size;
		this._pinCount = 0;
		this._state = state;
		this._data = data;
		this._retainHintCount = 0;
		this._referenceCount = 1;
		this._cacheMeta = null;
	}

	public BlockKey getKey() {
		return _key;
	}

	public long getSize() {
		return _size;
	}

	public Object getData() {
		if (_pinCount > 0)
			return _data;
		throw new IllegalStateException("Cannot get the data of an unpinned entry");
	}

	public Object getDataUnsafe() {
		return _data;
	}

	public void setDataUnsafe(Object data) {
		if(data != null && _data != null)
			throw new IllegalStateException("Cannot overwrite data");
		_data = data;
	}

	public BlockState getState() {
		return _state;
	}

	public boolean isPinned() {
		return _pinCount > 0;
	}

	public synchronized int getPinCount() {
		return _pinCount;
	}

	public synchronized int addReference() {
		return ++_referenceCount;
	}

	public synchronized int forget() {
		return --_referenceCount;
	}

	public synchronized int getReferenceCount() {
		return _referenceCount;
	}

	public Object getCacheMeta() {
		return _cacheMeta;
	}

	public void setCacheMeta(Object meta) {
		_cacheMeta = meta;
	}

	public synchronized void setState(BlockState state) {
		_state = state;
	}

	public synchronized void addRetainHint(int cnt) {
		_retainHintCount += cnt;
	}

	public synchronized void addRetainHint() {
		_retainHintCount++;
	}

	public synchronized void removeRetainHint() {
		if (_retainHintCount <= 0)
			return;
		_retainHintCount--;
	}

	public synchronized int getRetainHintCount() {
		return _retainHintCount;
	}

	/**
	 * Tries to clear the underlying data if it is not pinned
	 * @return the number of cleared bytes (or 0 if could not clear or data was already cleared)
	 */
	public synchronized long clear() {
		if (_pinCount != 0 || _data == null)
			return 0;
		_data = null;
		_retainHintCount = 0;
		return _size;
	}

	/**
	 * Pins the underlying data in memory
	 * @return the new number of pins (0 if pin was unsuccessful)
	 */
	public synchronized int pin() {
		if (_data == null)
			return 0;
		_pinCount++;
		return _pinCount;
	}

	/**
	 * Tries to increment pin-count if already pinned. Unpinned entries are not affected
	 * by this operation. This allows bypassing the global cache lock.
	 */
	public synchronized boolean fastPin() {
		if(_pinCount == 0)
			return false;
		_pinCount++;
		return true;
	}

	/**
	 * Unpins the underlying data
	 * @return true if the data is now unpinned
	 */
	public synchronized boolean unpin() {
		if (_pinCount <= 0)
			throw new IllegalStateException("Cannot unpin data if it was not pinned");
		_pinCount--;
		return _pinCount == 0;
	}

	/**
	 * Tries to unpin but guarantees that it will not
	 * remove the last pin. This allows bypassing the global cache lock.
	 */
	public synchronized boolean fastUnpin() {
		if(_pinCount <= 1)
			return false;
		_pinCount--;
		return true;
	}

	public String toString() {
		return "Entry" + _key.toString();
	}
}
