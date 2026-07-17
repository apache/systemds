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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class GlobalMemoryBroker implements MemoryBroker {
	private static final long RECLAIM_RETRY_DELAY_MS = 5;
	private static final double RECLAIM_PRESSURE = 0.85;
	private static final ScheduledThreadPoolExecutor RECLAIM_EXECUTOR = createReclaimExecutor();

	private enum BrokerMode {
		RELAXED, STRICT
	}

	private static final GlobalMemoryBroker BROKER = new GlobalMemoryBroker(Runtime.getRuntime().maxMemory() / 3);

	public static GlobalMemoryBroker get() {
		return BROKER;
	}

	private final long _allowedBytes;
	private final CopyOnWriteArrayList<MemoryAllowance> _allowances;
	private final AtomicBoolean _reclaimRunning;
	private long _usedBytes;
	private BrokerMode _brokerMode;

	private record TargetUpdate(MemoryAllowance _allowance, long _target) {}

	private static ScheduledThreadPoolExecutor createReclaimExecutor() {
		ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1, runnable -> {
			Thread thread = new Thread(runnable, "ooc-memory-broker-reclaimer");
			thread.setDaemon(true);
			return thread;
		});
		executor.setRemoveOnCancelPolicy(true);
		return executor;
	}

	public GlobalMemoryBroker(long allowedBytes) {
		_allowedBytes = allowedBytes;
		_usedBytes = 0;
		_allowances = new CopyOnWriteArrayList<>();
		_reclaimRunning = new AtomicBoolean(false);
	}

	@Override
	public long requestMemory(MemoryAllowance allowance, long minSize, long maxSize) {
		List<TargetUpdate> updates = null;
		long allow = 0;
		synchronized(this) {
			if(minSize < 0 || maxSize < minSize)
				throw new IllegalArgumentException();
			long free = _allowedBytes - _usedBytes;
			if(free >= minSize) {
				allow = Math.min(free, maxSize);
				_usedBytes += allow;
				updates = rebalance(false);
			}
		}
		if(updates != null)
			applyTargetUpdates(updates);
		return allow;
	}

	@Override
	public void freeMemory(MemoryAllowance allowance, long freedMemory) {
		List<TargetUpdate> updates;
		synchronized(this) {
			if(freedMemory < 0)
				throw new IllegalArgumentException();
			_usedBytes -= freedMemory;
			updates = rebalanceAfterFree();
		}
		if(updates != null)
			applyTargetUpdates(updates);
		if(freedMemory > 0)
			notifyReservationWaiters();
	}

	@Override
	public void shutdownAllowance(MemoryAllowance allowance) {
		List<TargetUpdate> updates;
		synchronized(this) {
			updates = rebalance(true);
		}
		applyTargetUpdates(updates);
		notifyReservationWaiters();
	}

	@Override
	public void destroyAllowance(MemoryAllowance allowance, long freedMemory) {
		List<TargetUpdate> updates;
		synchronized(this) {
			if(freedMemory < 0)
				throw new IllegalArgumentException();
			_allowances.remove(allowance);
			_usedBytes -= freedMemory;
			updates = rebalance(true);
		}
		applyTargetUpdates(updates);
		notifyReservationWaiters();
	}

	@Override
	public synchronized void attachAllowance(MemoryAllowance allowance) {
		_allowances.add(allowance);
		allowance.setTargetMemory(_allowedBytes);
	}

	@Override
	public void reservationBlocked(MemoryAllowance allowance, long bytes) {
		if(_reclaimRunning.compareAndSet(false, true))
			RECLAIM_EXECUTOR.execute(this::runReclaim);
	}

	private void runReclaim() {
		try {
			long reclaimed = 0;
			for(MemoryAllowance allowance : _allowances)
				if(!allowance.isShutdown())
					reclaimed += allowance.reclaimUnused();
			if(reclaimed == 0)
				return;

			List<TargetUpdate> updates;
			synchronized(this) {
				_usedBytes = Math.max(0, _usedBytes - reclaimed);
				updates = rebalanceAfterFree();
			}
			if(updates != null)
				applyTargetUpdates(updates);
			notifyReservationWaiters();
		}
		finally {
			if(shouldRetryReclaim())
				RECLAIM_EXECUTOR.schedule(this::runReclaim, RECLAIM_RETRY_DELAY_MS, TimeUnit.MILLISECONDS);
			else {
				_reclaimRunning.set(false);
				if(shouldRetryReclaim() && _reclaimRunning.compareAndSet(false, true))
					RECLAIM_EXECUTOR.execute(this::runReclaim);
			}
		}
	}

	private boolean shouldRetryReclaim() {
		if(!hasReclaimPressure())
			return false;
		for(MemoryAllowance allowance : _allowances) {
			if(allowance instanceof SyncMemoryAllowance sync && sync.hasReservationWaiters())
				return true;
		}
		return false;
	}

	private synchronized boolean hasReclaimPressure() {
		return _usedBytes >= _allowedBytes * RECLAIM_PRESSURE;
	}

	private List<TargetUpdate> rebalance(boolean force) {
		long free = _allowedBytes - _usedBytes;
		if(force)
			_brokerMode = null;
		if(free > _allowedBytes / 5)
			return switchBrokerMode(BrokerMode.RELAXED);
		else
			return switchBrokerMode(BrokerMode.STRICT);
	}

	private List<TargetUpdate> rebalanceAfterFree() {
		long free = _allowedBytes - _usedBytes;
		if(_brokerMode == BrokerMode.RELAXED && free > _allowedBytes / 5)
			return rebalanceToRelaxed();
		return rebalance(false);
	}

	private List<TargetUpdate> switchBrokerMode(BrokerMode newMode) {
		if(newMode == _brokerMode)
			return null;
		List<TargetUpdate> updates = switch(newMode) {
			case STRICT -> rebalanceToStrict();
			case RELAXED -> rebalanceToRelaxed();
			default -> throw new IllegalStateException("Unsupported broker mode " + newMode);
		};
		_brokerMode = newMode;
		return updates;
	}

	private List<TargetUpdate> rebalanceToStrict() {
		List<TargetUpdate> updates = new ArrayList<>();
		long share = getEqualShare();
		for(MemoryAllowance allowance : _allowances) {
			if(allowance.isShutdown())
				continue;
			if(allowance.getUsedMemory() > share) {
				updates.add(new TargetUpdate(allowance,
					Math.min(allowance.getTargetMemory(), share + (long) ((allowance.getUsedMemory() - share) * 0.9))));
			}
		}
		return updates;
	}

	private List<TargetUpdate> rebalanceToRelaxed() {
		List<TargetUpdate> updates = new ArrayList<>();
		long free = _allowedBytes - _usedBytes;
		for(MemoryAllowance allowance : _allowances) {
			if(allowance.isShutdown())
				continue;
			updates.add(new TargetUpdate(allowance, allowance.getGrantedMemory() + free));
		}
		return updates;
	}

	private long getEqualShare() {
		int active = 0;
		for(MemoryAllowance allowance : _allowances)
			if(!allowance.isShutdown())
				active++;
		return active == 0 ? _allowedBytes : _allowedBytes / active;
	}

	private void notifyReservationWaiters() {
		for(MemoryAllowance allowance : _allowances) {
			if(allowance instanceof SyncMemoryAllowance sync)
				sync.onBrokerMemoryAvailable();
		}
	}

	private static void applyTargetUpdates(List<TargetUpdate> updates) {
		for(TargetUpdate update : updates)
			update._allowance.setTargetMemory(update._target);
	}
}
