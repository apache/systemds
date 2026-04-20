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

public class SyncMemoryAllowance implements MemoryAllowance {
	protected final MemoryBroker _broker;
	protected volatile long _usedBytes;
	protected volatile long _grantedBytes;
	protected volatile long _targetBytes;
	protected volatile boolean _shutdown;
	protected volatile boolean _destroyed;

	public SyncMemoryAllowance(MemoryBroker broker) {
		_broker = broker;
		_usedBytes = 0;
		_grantedBytes = 0;
		_targetBytes = 0;
		_shutdown = false;
		_destroyed = false;
		broker.attachAllowance(this);
	}

	@Override
	public boolean tryReserve(long bytes) {
		long minRequest;
		long maxRequest;
		synchronized(this) {
			if(_shutdown || _destroyed)
				return false;
			if(_usedBytes + bytes > _targetBytes)
				return false;
			if(_usedBytes + bytes <= _grantedBytes) {
				_usedBytes += bytes;
				return true;
			}
			minRequest = _usedBytes + bytes - _grantedBytes;
			maxRequest = Math.max(minRequest, Math.max(_grantedBytes, bytes) * 2);
		}

		long granted = _broker.requestMemory(this, minRequest, maxRequest);
		long refund = 0;
		boolean success = false;
		synchronized(this) {
			if(_shutdown || _destroyed)
				refund = granted;
			else {
				_grantedBytes += granted;
				if(_usedBytes + bytes <= _targetBytes && _usedBytes + bytes <= _grantedBytes) {
					_usedBytes += bytes;
					success = true;
				}
				notifyAll();
			}
		}
		if(refund > 0)
			_broker.freeMemory(this, refund);
		return success;
	}

	@Override
	public void reserveBlocking(long bytes) {
		if(_shutdown || _destroyed)
			throw new IllegalStateException("Cannot reserve memory on closed allowance.");
		while(true) {
			if(tryReserve(bytes)) {
				synchronized(this) {
					notifyAll();
				}
				return;
			}
			synchronized(this) {
				if(_shutdown || _destroyed)
					throw new IllegalStateException("Cannot reserve memory on closed allowance.");
				try {
					wait();
				}
				catch(InterruptedException e) {
					throw new DMLRuntimeException(e);
				}
			}
		}
	}

	@Override
	public void release(long bytes) {
		long freedMemory = 0;
		long destroyFreedMemory = 0;
		boolean destroy = false;
		synchronized(this) {
			_usedBytes -= bytes;
			if(_shutdown) {
				long oldGrantedBytes = _grantedBytes;
				_grantedBytes = _usedBytes;
				if(_usedBytes == 0) {
					_destroyed = true;
					destroy = true;
					destroyFreedMemory = oldGrantedBytes;
				}
				else {
					freedMemory = oldGrantedBytes - _grantedBytes;
				}
			}
			else if(_grantedBytes > _targetBytes) {
				long oldGrantedBytes = _grantedBytes;
				_grantedBytes = Math.max(_usedBytes, _targetBytes);
				freedMemory = oldGrantedBytes - _grantedBytes;
			}
			notifyAll();
		}
		if(destroy)
			_broker.destroyAllowance(this, destroyFreedMemory);
		else if(freedMemory > 0)
			_broker.freeMemory(this, freedMemory);
	}

	@Override
	public long getUsedMemory() {
		return _usedBytes;
	}

	@Override
	public long getGrantedMemory() {
		return _grantedBytes;
	}

	@Override
	public long getTargetMemory() {
		return _targetBytes;
	}

	@Override
	public synchronized void setTargetMemory(long targetMemory) {
		if(_shutdown || _destroyed)
			return;
		_targetBytes = targetMemory;
		notifyAll();
	}

	@Override
	public void shutdown() {
		long freedMemory = 0;
		long destroyFreedMemory = 0;
		boolean destroy = false;
		synchronized(this) {
			if(_shutdown || _destroyed)
				return;
			_shutdown = true;
			long oldGrantedBytes = _grantedBytes;
			_grantedBytes = _usedBytes;
			_targetBytes = 0;
			if(_usedBytes == 0) {
				_destroyed = true;
				destroy = true;
				destroyFreedMemory = oldGrantedBytes;
			}
			else {
				freedMemory = oldGrantedBytes - _grantedBytes;
			}
			notifyAll();
		}
		_broker.shutdownAllowance(this);
		if(destroy)
			_broker.destroyAllowance(this, destroyFreedMemory);
		else if(freedMemory > 0)
			_broker.freeMemory(this, freedMemory);
	}

	@Override
	public boolean isShutdown() {
		return _shutdown || _destroyed;
	}
}
