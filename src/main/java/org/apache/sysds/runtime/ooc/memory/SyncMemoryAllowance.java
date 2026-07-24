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
import org.apache.sysds.runtime.ooc.cache.OOCFuture;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;

public class SyncMemoryAllowance implements MemoryAllowance {
	private static final long RELEASE_TRIM_BUFFER_BYTES = 20_000_000L;

	protected final MemoryBroker _broker;
	protected final long _consumptionLimit;
	protected final long _minimumOperatingBytes;
	protected volatile long _usedBytes;
	protected volatile long _grantedBytes;
	protected volatile long _targetBytes;
	protected volatile boolean _shutdown;
	protected volatile boolean _destroyed;
	private final Queue<ReservationWaiter> _reservationWaiters;
	private boolean _drainingReservationWaiters;
	private boolean _reservationDrainRequested;

	public SyncMemoryAllowance(MemoryBroker broker) {
		this(broker, Long.MAX_VALUE);
	}

	public SyncMemoryAllowance(MemoryBroker broker, long consumptionLimit) {
		this(broker, consumptionLimit, 0);
	}

	public SyncMemoryAllowance(MemoryBroker broker, long consumptionLimit, long minimumOperatingBytes) {
		if(consumptionLimit < 0)
			throw new IllegalArgumentException("Consumption limit must not be negative: " + consumptionLimit);
		if(minimumOperatingBytes < 0)
			throw new IllegalArgumentException(
				"Minimum operating memory must not be negative: " + minimumOperatingBytes);
		_broker = broker;
		_consumptionLimit = consumptionLimit;
		_minimumOperatingBytes = Math.min(minimumOperatingBytes, consumptionLimit);
		_usedBytes = 0;
		_grantedBytes = 0;
		_targetBytes = 0;
		_shutdown = false;
		_destroyed = false;
		_reservationWaiters = new ConcurrentLinkedQueue<>();
		_drainingReservationWaiters = false;
		_reservationDrainRequested = false;
		broker.attachAllowance(this);
	}

	@Override
	public boolean tryReserve(long bytes) {
		if(bytes < 0)
			throw new IllegalArgumentException("Cannot reserve negative bytes: " + bytes);
		if(bytes > _consumptionLimit)
			throw new IllegalArgumentException("Cannot reserve more memory than the consumption limit");
		long minRequest;
		long maxRequest;
		synchronized(this) {
			if(_shutdown || _destroyed)
				return false;
			if(_usedBytes + bytes <= _grantedBytes && _usedBytes + bytes <= _targetBytes) {
				_usedBytes += bytes;
				return true;
			}
			if(_usedBytes + bytes > _targetBytes)
				return false;
			minRequest = _usedBytes + bytes - _grantedBytes;
			maxRequest = Math.max(minRequest, Math.max(_grantedBytes, bytes) * 2);
		}

		long granted = _broker.requestMemory(this, minRequest, maxRequest);
		long refund = 0;
		boolean success = false;
		boolean drainWaiters = false;
		synchronized(this) {
			if(_shutdown || _destroyed)
				refund = granted;
			else {
				_grantedBytes += granted;
				if(_usedBytes + bytes <= _targetBytes && _usedBytes + bytes <= _grantedBytes) {
					_usedBytes += bytes;
					success = true;
				}
				drainWaiters = success && !_reservationWaiters.isEmpty();
				notifyAll();
			}
		}
		if(refund > 0)
			_broker.freeMemory(this, refund);
		if(drainWaiters)
			requestReservationDrain();
		return success;
	}

	@Override
	public void reserveBlocking(long bytes) {
		try {
			reserveAsync(bytes).get();
		}
		catch(InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new DMLRuntimeException(e);
		}
		catch(ExecutionException e) {
			throw DMLRuntimeException.of(e.getCause());
		}
	}

	@Override
	public OOCFuture<Void> reserveAsync(long bytes) {
		if(bytes < 0)
			throw new IllegalArgumentException("Cannot reserve negative bytes: " + bytes);
		if(bytes == 0)
			return OOCFuture.completed(null);
		if(bytes > _consumptionLimit)
			return OOCFuture
				.failed(new IllegalArgumentException("Cannot reserve more memory than the consumption limit"));
		if(_shutdown || _destroyed)
			return OOCFuture.failed(new IllegalStateException("Cannot reserve memory on closed allowance."));
		if(_reservationWaiters.isEmpty() && tryReserve(bytes))
			return OOCFuture.completed(null);
		OOCFuture<Void> future = new OOCFuture<>();
		ReservationWaiter waiter = new ReservationWaiter(bytes, future);
		_reservationWaiters.add(waiter);
		if((_shutdown || _destroyed) && _reservationWaiters.remove(waiter)) {
			future.completeExceptionally(new IllegalStateException("Cannot reserve memory on closed allowance."));
			return future;
		}
		requestReservationDrain();
		if(!future.isDone())
			_broker.reservationBlocked(this, bytes);
		return future;
	}

	@Override
	public void release(long bytes) {
		long freedMemory = 0;
		long destroyFreedMemory = 0;
		boolean destroy = false;
		boolean drainWaiters;
		synchronized(this) {
			if(bytes < 0)
				throw new IllegalArgumentException("Cannot release negative bytes: " + bytes);
			if(_usedBytes < bytes) {
				throw new IllegalArgumentException("Memory allowance underflow in " + getClass().getSimpleName()
					+ ": release=" + bytes + ", used=" + _usedBytes + ", granted=" + _grantedBytes + ", target="
					+ _targetBytes + ", shutdown=" + _shutdown + ", destroyed=" + _destroyed);
			}
			_usedBytes -= bytes;
			if(_shutdown) {
				long oldGrantedBytes = _grantedBytes;
				_grantedBytes = _usedBytes;
				if(_grantedBytes < 0) {
					throw new IllegalArgumentException("Granted memory underflow in " + getClass().getSimpleName()
						+ ": granted=" + _grantedBytes + ", used=" + _usedBytes + ", released=" + bytes);
				}
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
			else if(_usedBytes * 3 < _grantedBytes * 2) {
				long oldGrantedBytes = _grantedBytes;
				_grantedBytes = Math.max(_usedBytes, Math.min(_grantedBytes, _usedBytes + RELEASE_TRIM_BUFFER_BYTES));
				freedMemory = oldGrantedBytes - _grantedBytes;
			}
			drainWaiters = !_reservationWaiters.isEmpty() && !_shutdown && !_destroyed;
			notifyAll();
		}
		if(destroy)
			_broker.destroyAllowance(this, destroyFreedMemory);
		else if(freedMemory > 0)
			_broker.freeMemory(this, freedMemory);
		if(drainWaiters)
			requestReservationDrain();
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
	public void setTargetMemory(long targetMemory) {
		if(targetMemory < 0)
			throw new IllegalArgumentException("Target memory must not be negative: " + targetMemory);
		long freedMemory = 0;
		boolean drainWaiters = false;
		synchronized(this) {
			if(_shutdown || _destroyed)
				return;
			_targetBytes = Math.min(Math.max(targetMemory, _minimumOperatingBytes), _consumptionLimit);
			if(_grantedBytes > _targetBytes) {
				long oldGrantedBytes = _grantedBytes;
				_grantedBytes = Math.max(_usedBytes, _targetBytes);
				freedMemory = oldGrantedBytes - _grantedBytes;
			}
			drainWaiters = !_reservationWaiters.isEmpty();
			notifyAll();
		}
		if(freedMemory > 0)
			_broker.freeMemory(this, freedMemory);
		if(drainWaiters)
			requestReservationDrain();
	}

	@Override
	public synchronized long reclaimUnused() {
		if(_shutdown || _destroyed || _grantedBytes <= _usedBytes)
			return 0;
		long reclaimed = _grantedBytes - _usedBytes;
		_grantedBytes = _usedBytes;
		notifyAll();
		return reclaimed;
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
		IllegalStateException ex = new IllegalStateException("Cannot reserve memory on closed allowance.");
		ReservationWaiter waiter;
		while((waiter = _reservationWaiters.poll()) != null)
			waiter.future.completeExceptionally(ex);
	}

	@Override
	public boolean isShutdown() {
		return _shutdown || _destroyed;
	}

	void onBrokerMemoryAvailable() {
		boolean drainWaiters;
		synchronized(this) {
			drainWaiters = !_reservationWaiters.isEmpty() && !_shutdown && !_destroyed;
			notifyAll();
		}
		if(drainWaiters)
			requestReservationDrain();
	}

	boolean hasReservationWaiters() {
		return !_reservationWaiters.isEmpty();
	}

	private void requestReservationDrain() {
		synchronized(this) {
			_reservationDrainRequested = true;
			if(_drainingReservationWaiters)
				return;
			_drainingReservationWaiters = true;
		}
		try {
			while(true) {
				synchronized(this) {
					_reservationDrainRequested = false;
				}
				drainReservationWaitersOnce();
				synchronized(this) {
					if(!_reservationDrainRequested) {
						_drainingReservationWaiters = false;
						return;
					}
				}
			}
		}
		catch(RuntimeException | Error t) {
			synchronized(this) {
				_drainingReservationWaiters = false;
			}
			throw t;
		}
	}

	private void drainReservationWaitersOnce() {
		while(true) {
			if(_shutdown || _destroyed)
				return;
			ReservationWaiter waiter = _reservationWaiters.peek();
			if(waiter == null)
				return;
			boolean admitted;
			try {
				admitted = tryReserve(waiter.bytes);
			}
			catch(Throwable t) {
				removeReservationWaiter(waiter);
				waiter.future.completeExceptionally(t);
				continue;
			}
			if(!admitted)
				return;
			if(removeReservationWaiter(waiter))
				waiter.future.complete(null);
			else
				release(waiter.bytes);
		}
	}

	private boolean removeReservationWaiter(ReservationWaiter waiter) {
		return _reservationWaiters.remove(waiter);
	}

	private record ReservationWaiter(long bytes, OOCFuture<Void> future) {
	}
}
