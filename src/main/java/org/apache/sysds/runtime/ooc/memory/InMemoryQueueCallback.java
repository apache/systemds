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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

import java.util.concurrent.atomic.AtomicInteger;

public class InMemoryQueueCallback implements OOCStream.QueueCallback<IndexedMatrixValue> {
	private CallbackHandle _handle;
	private boolean _closed;

	public InMemoryQueueCallback(IndexedMatrixValue result, DMLRuntimeException failure, MemoryAllowance allow,
		long reservedBytes) {
		_handle = new CallbackHandle(result, failure, allow, reservedBytes);
		_closed = false;
	}

	private InMemoryQueueCallback(CallbackHandle handle) {
		_handle = handle;
		_closed = false;
	}

	@Override
	public IndexedMatrixValue get() {
		return _handle.get();
	}

	@Override
	public synchronized OOCStream.QueueCallback<IndexedMatrixValue> keepOpen() {
		if(_closed)
			throw new IllegalStateException("Cannot keep open a closed callback");
		_handle._refCtr.incrementAndGet();
		return new InMemoryQueueCallback(_handle);
	}

	@Override
	public void fail(DMLRuntimeException failure) {
		_handle._failure = failure;
	}

	public long getManagedBytes() {
		synchronized(_handle) {
			return _handle._reservedBytes;
		}
	}

	public boolean tryTransferOwnership(MemoryAllowance allowance) {
		synchronized(_handle) {
			long bytes = _handle._reservedBytes;
			if(bytes <= 0 || _handle._allow == allowance)
				return true;
			if(_handle._cacheIdx >= 0)
				return false;
			if(!allowance.tryReserve(bytes))
				return false;
			_handle._allow.release(bytes);
			_handle._allow = allowance;
			return true;
		}
	}

	public void transferOwnershipBlocking(MemoryAllowance allowance) {
		synchronized(_handle) {
			long bytes = _handle._reservedBytes;
			if(bytes <= 0 || _handle._allow == allowance)
				return;
			if(_handle._cacheIdx >= 0)
				throw new IllegalStateException("Cannot transfer ownership of a cached allowance callback.");
			if(allowance instanceof CachedAllowance cached)
				cached.admitBlocking(bytes);
			else
				allowance.reserveBlocking(bytes);
			_handle._allow.release(bytes);
			_handle._allow = allowance;
		}
	}

	public long releaseManagedMemory() {
		synchronized(_handle) {
			long bytes = _handle._reservedBytes;
			if(bytes <= 0)
				return 0;
			_handle._reservedBytes = 0;
			_handle._allow.release(bytes);
			return bytes;
		}
	}

	@Override
	public synchronized void close() {
		if(_closed)
			return;
		_closed = true;
		if(_handle._refCtr.decrementAndGet() == 0)
			_handle.closeFinal();
		_handle = null;
	}

	@Override
	public boolean isEos() {
		return _handle.isEos();
	}

	@Override
	public boolean isFailure() {
		return _handle._failure != null;
	}

	CallbackHandle getHandle() {
		return _handle;
	}

	static final class CallbackHandle {
		private volatile IndexedMatrixValue _result;
		private final AtomicInteger _refCtr;
		private MemoryAllowance _allow;
		private long _reservedBytes;
		private volatile DMLRuntimeException _failure;
		private int _cacheIdx;

		private CallbackHandle(IndexedMatrixValue result, DMLRuntimeException failure, MemoryAllowance allow,
			long reservedBytes) {
			_result = result;
			_failure = failure;
			_refCtr = new AtomicInteger(1);
			_allow = allow;
			_reservedBytes = reservedBytes;
			_cacheIdx = -1;
		}

		private IndexedMatrixValue get() {
			if(_failure != null)
				throw _failure;
			return _result;
		}

		private boolean isEos() {
			return _failure == null && _result == null;
		}

		synchronized void attachCachedAllowance(CachedAllowance allowance, int index) {
			if(_allow != allowance)
				throw new IllegalStateException("Callback ownership must already belong to the cached allowance.");
			if(_cacheIdx >= 0 && _cacheIdx != index)
				throw new IllegalStateException("Callback is already attached to a different cached slot.");
			_cacheIdx = index;
		}

		synchronized void detachCachedAllowance() {
			_cacheIdx = -1;
		}

		boolean isExclusiveToRoot() {
			return _refCtr.get() == 1;
		}

		private synchronized IndexedMatrixValue takeManagedResultForHandover() {
			IndexedMatrixValue result = _result;
			_result = null;
			return result;
		}

		private void closeFinal() {
			_result = null;
			_allow.release(_reservedBytes);
			_reservedBytes = 0;
			_cacheIdx = -1;
		}
	}

	public IndexedMatrixValue takeManagedResultForHandover() {
		return _handle.takeManagedResultForHandover();
	}
}
