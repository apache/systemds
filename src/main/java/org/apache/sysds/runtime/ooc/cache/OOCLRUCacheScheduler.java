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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.utils.Statistics;
import scala.Tuple2;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

public class OOCLRUCacheScheduler implements OOCCacheScheduler {
	private static final boolean SANITY_CHECKS = false;

	private final OOCIOHandler _ioHandler;
	private final LinkedHashMap<BlockKey, BlockEntry> _cache;
	private final HashMap<BlockKey, BlockEntry> _evictionCache;
	private final Deque<DeferredReadRequest> _deferredReadRequests;
	private final Deque<DeferredReadRequest> _processingReadRequests;
	private final long _hardLimit;
	private final long _evictionLimit;
	private final int _callerId;
	private long _cacheSize;
	private long _bytesUpForEviction;
	private volatile boolean _running;
	private boolean _warnThrottling;

	public OOCLRUCacheScheduler(OOCIOHandler ioHandler, long evictionLimit, long hardLimit) {
		this._ioHandler = ioHandler;
		this._cache = new LinkedHashMap<>(1024, 0.75f, true);
		this._evictionCache = new  HashMap<>();
		this._deferredReadRequests = new ArrayDeque<>();
		this._processingReadRequests = new ArrayDeque<>();
		this._hardLimit = hardLimit;
		this._evictionLimit = evictionLimit;
		this._cacheSize = 0;
		this._bytesUpForEviction = 0;
		this._running = true;
		this._warnThrottling = false;
		this._callerId = DMLScript.OOC_LOG_EVENTS ? OOCEventLog.registerCaller("LRUCacheScheduler") : 0;

		if (DMLScript.OOC_LOG_EVENTS) {
			OOCEventLog.putRunSetting("CacheEvictionLimit", _evictionLimit);
			OOCEventLog.putRunSetting("CacheHardLimit", _hardLimit);
		}
	}

	@Override
	public CompletableFuture<BlockEntry> request(BlockKey key) {
		if (!this._running)
			throw new IllegalStateException("Cache scheduler has been shut down.");

		Statistics.incrementOOCEvictionGet();

		BlockEntry entry;
		boolean couldPin = false;
		synchronized(this) {
			entry = _cache.get(key);
			if (entry == null)
				entry = _evictionCache.get(key);
			if (entry == null)
				throw new IllegalArgumentException("Could not find requested block with key " + key);

			synchronized(entry) {
				if (entry.getState().isAvailable()) {
					if (entry.pin() == 0)
						throw new IllegalStateException();
					couldPin = true;
				}
			}
		}

		if (couldPin) {
			// Then we could pin the required entry and can terminate
			return CompletableFuture.completedFuture(entry);
		}

		//System.out.println("Requesting deferred: " + key);
		// Schedule deferred read otherwise
		final CompletableFuture<BlockEntry> future = new CompletableFuture<>();
		final CompletableFuture<List<BlockEntry>> requestFuture = new CompletableFuture<>();
		requestFuture.whenComplete((r, t) -> future.complete(r.get(0)));
		scheduleDeferredRead(new DeferredReadRequest(requestFuture, Collections.singletonList(entry)));
		return future;
	}

	@Override
	public CompletableFuture<List<BlockEntry>> request(List<BlockKey> keys) {
		if (!this._running)
			throw new IllegalStateException("Cache scheduler has been shut down.");

		Statistics.incrementOOCEvictionGet(keys.size());

		List<BlockEntry> entries = new ArrayList<>(keys.size());
		boolean couldPinAll = true;

		synchronized(this) {
			for (BlockKey key : keys) {
				BlockEntry entry = _cache.get(key);
				if (entry == null)
					entry = _evictionCache.get(key);
				if (entry == null)
					throw new IllegalArgumentException("Could not find requested block with key " + key);

				if (couldPinAll) {
					synchronized(entry) {
						if(entry.getState().isAvailable()) {
							if(entry.pin() == 0)
								throw new IllegalStateException();
						}
						else {
							couldPinAll = false;
						}
					}

					if (!couldPinAll) {
						// Undo pin for all previous entries
						for (BlockEntry e : entries)
							e.unpin(); // Do not unpin using unpin(...) method to avoid explicit eviction on memory pressure
					}
				}
				entries.add(entry);
			}
		}

		if (couldPinAll) {
			// Then we could pin all entries
			return CompletableFuture.completedFuture(entries);
		}

		// Schedule deferred read otherwise
		final  CompletableFuture<List<BlockEntry>> future = new CompletableFuture<>();
		scheduleDeferredRead(new DeferredReadRequest(future, entries));
		return future;
	}

	private void scheduleDeferredRead(DeferredReadRequest deferredReadRequest) {
		synchronized(this) {
			_deferredReadRequests.add(deferredReadRequest);
		}
		onCacheSizeChanged(false); // To schedule deferred reads if possible
	}

	@Override
	public void put(BlockKey key, Object data, long size) {
		put(key, data, size, false);
	}

	@Override
	public BlockEntry putAndPin(BlockKey key, Object data, long size) {
		return put(key, data, size, true);
	}

	private BlockEntry put(BlockKey key, Object data, long size, boolean pin) {
		if (!this._running)
			throw new IllegalStateException();
		if (data == null)
			throw new IllegalArgumentException();

		Statistics.incrementOOCEvictionPut();
		BlockEntry entry = new BlockEntry(key, size, data);
		if (pin)
			entry.pin();
		synchronized(this) {
			BlockEntry avail = _cache.putIfAbsent(key, entry);
			if (avail != null || _evictionCache.containsKey(key))
				throw new IllegalStateException("Cannot overwrite existing entries: " + key);
			_cacheSize += size;
		}
		onCacheSizeChanged(true);
		return entry;
	}

	@Override
	public void forget(BlockKey key) {
		if (!this._running)
			return;
		BlockEntry entry;
		boolean shouldScheduleDeletion = false;
		long cacheSizeDelta = 0;
		synchronized(this) {
			entry = _cache.remove(key);

			if (entry == null)
				entry = _evictionCache.remove(key);

			if (entry != null) {
				synchronized(entry) {
					shouldScheduleDeletion = entry.getState().isBackedByDisk()
						|| entry.getState() == BlockState.EVICTING;
					cacheSizeDelta = transitionMemState(entry, BlockState.REMOVED);
				}

			}
		}
		if (cacheSizeDelta != 0)
			onCacheSizeChanged(cacheSizeDelta > 0);
		if (shouldScheduleDeletion)
			_ioHandler.scheduleDeletion(entry);
	}

	@Override
	public void pin(BlockEntry entry) {
		if (!this._running)
			throw new IllegalStateException("Cache scheduler has been shut down.");

		int pinCount = entry.pin();
		if (pinCount == 0)
			throw new IllegalStateException("Could not pin the requested entry: " + entry.getKey());
		synchronized(this) {
			// Access element in cache for Lru
			_cache.get(entry.getKey());
		}
	}

	@Override
	public void unpin(BlockEntry entry) {
		boolean couldFree = entry.unpin();

		if (couldFree) {
			long cacheSizeDelta = 0;
			synchronized(this) {
				if (_cacheSize <= _evictionLimit)
					return; // Nothing to do

				synchronized(entry) {
					if (entry.isPinned())
						return; // Pin state changed so we cannot evict

					if (entry.getState().isAvailable() && entry.getState().isBackedByDisk()) {
						cacheSizeDelta =  transitionMemState(entry, BlockState.COLD);
						long cleared = entry.clear();
						if (cleared != entry.getSize())
							throw new IllegalStateException();
						_cache.remove(entry.getKey());
						_evictionCache.put(entry.getKey(), entry);
					} else if (entry.getState() == BlockState.HOT) {
						cacheSizeDelta = onUnpinnedHotBlockUnderMemoryPressure(entry);
					}
				}
			}
			if (cacheSizeDelta != 0)
				onCacheSizeChanged(cacheSizeDelta > 0);
		}
	}

	@Override
	public synchronized void shutdown() {
		this._running = false;
		_cache.clear();
		_evictionCache.clear();
		_processingReadRequests.clear();
		_deferredReadRequests.clear();
		_cacheSize = 0;
		_bytesUpForEviction = 0;
	}

	/**
	 * Must be called while this cache and the corresponding entry are locked
	 */
	private long onUnpinnedHotBlockUnderMemoryPressure(BlockEntry entry) {
		long cacheSizeDelta = transitionMemState(entry, BlockState.EVICTING);
		evict(entry);
		return cacheSizeDelta;
	}

	private void onCacheSizeChanged(boolean incr) {
		if (incr)
			onCacheSizeIncremented();
		else {
			while(onCacheSizeDecremented()) {}
		}
		if (DMLScript.OOC_LOG_EVENTS)
			OOCEventLog.onCacheSizeChangedEvent(_callerId, System.nanoTime(), _cacheSize, _bytesUpForEviction);
	}

	private synchronized void sanityCheck() {
		if (_cacheSize > _hardLimit) {
			if (!_warnThrottling) {
				_warnThrottling = true;
				System.out.println("[INFO] Throttling: " + _cacheSize/1000 + "KB - " + _bytesUpForEviction/1000 + "KB > " + _hardLimit/1000 + "KB");
			}
		}
		else if (_warnThrottling) {
			_warnThrottling = false;
			System.out.println("[INFO] No more throttling: " + _cacheSize/1000 + "KB - " + _bytesUpForEviction/1000 + "KB <= " + _hardLimit/1000 + "KB");
		}

		if (!SANITY_CHECKS)
			return;

		int pinned = 0;
		int backedByDisk = 0;
		int evicting = 0;
		int total = 0;
		long actualCacheSize = 0;
		long upForEviction = 0;
		for (BlockEntry entry : _cache.values()) {
			if (entry.isPinned())
				pinned++;
			if (entry.getState().isBackedByDisk())
				backedByDisk++;
			if (entry.getState() == BlockState.EVICTING) {
				evicting++;
				upForEviction += entry.getSize();
			}
			if (!entry.getState().isAvailable())
				throw new IllegalStateException();
			total++;
			actualCacheSize += entry.getSize();
		}
		for (BlockEntry entry : _evictionCache.values()) {
			if (entry.getState().isAvailable())
				throw new IllegalStateException("Invalid eviction state: " + entry.getState());
			if (entry.getState() == BlockState.READING)
				actualCacheSize += entry.getSize();
		}
		if (actualCacheSize != _cacheSize)
			throw new IllegalStateException(actualCacheSize + " != " + _cacheSize);
		if (upForEviction != _bytesUpForEviction)
			throw new IllegalStateException(upForEviction + " != " + _bytesUpForEviction);
		System.out.println("==========");
		System.out.println("Limit: " + _evictionLimit/1000 + "KB");
		System.out.println("Memory: (" + _cacheSize/1000 + "KB - " + _bytesUpForEviction/1000 + "KB) / " + _hardLimit/1000 + "KB");
		System.out.println("Pinned: " + pinned + " / " + total);
		System.out.println("Disk backed: " + backedByDisk + " / " + total);
		System.out.println("Evicting: " + evicting + " / " + total);
	}

	private void onCacheSizeIncremented() {
		long cacheSizeDelta = 0;
		List<BlockEntry> upForEviction;
		synchronized(this) {
			if(_cacheSize - _bytesUpForEviction <= _evictionLimit)
				return; // Nothing to do

			// Scan for values that can be evicted
			Collection<BlockEntry> entries = _cache.values();
			List<BlockEntry> toRemove = new ArrayList<>();
			upForEviction = new ArrayList<>();

			for(BlockEntry entry : entries) {
				if(_cacheSize - _bytesUpForEviction <= _evictionLimit)
					break;

				synchronized(entry) {
					if(!entry.isPinned() && entry.getState().isBackedByDisk()) {
						cacheSizeDelta += transitionMemState(entry, BlockState.COLD);
						entry.clear();
						toRemove.add(entry);
					}
					else if(entry.getState() != BlockState.EVICTING && !entry.getState().isBackedByDisk()) {
						cacheSizeDelta += transitionMemState(entry, BlockState.EVICTING);
						upForEviction.add(entry);
					}
				}
			}

			for(BlockEntry entry : toRemove) {
				_cache.remove(entry.getKey());
				_evictionCache.put(entry.getKey(), entry);
			}

			sanityCheck();
		}

		for (BlockEntry entry : upForEviction) {
			evict(entry);
		}

		if (cacheSizeDelta != 0)
			onCacheSizeChanged(cacheSizeDelta > 0);
	}

	private boolean onCacheSizeDecremented() {
		boolean allReserved = true;
		List<Tuple2<Integer, BlockEntry>> toRead;
		DeferredReadRequest req;
		synchronized(this) {
			if(_cacheSize >= _hardLimit || _deferredReadRequests.isEmpty())
				return false; // Nothing to do

			// Try to schedule the next disk read
			req = _deferredReadRequests.peek();
			toRead = new ArrayList<>(req.getEntries().size());

			for(int idx = 0; idx < req.getEntries().size(); idx++) {
				if(!req.actionRequired(idx))
					continue;

				BlockEntry entry = req.getEntries().get(idx);
				synchronized(entry) {
					if(entry.getState().isAvailable()) {
						if(entry.pin() == 0)
							throw new IllegalStateException();
						req.setPinned(idx);
					}
					else {
						if(_cacheSize + entry.getSize() <= _hardLimit) {
							transitionMemState(entry, BlockState.READING);
							toRead.add(new Tuple2<>(idx, entry));
							req.schedule(idx);
						}
						else {
							allReserved = false;
						}
					}
				}
			}

			if (allReserved) {
				_deferredReadRequests.poll();
				if (!toRead.isEmpty())
					_processingReadRequests.add(req);
			}

			sanityCheck();
		}

		if (allReserved && toRead.isEmpty()) {
			req.getFuture().complete(req.getEntries());
			return true;
		}

		for (Tuple2<Integer, BlockEntry> tpl : toRead) {
			final int idx = tpl._1;
			final BlockEntry entry = tpl._2;
			CompletableFuture<BlockEntry> future = _ioHandler.scheduleRead(entry);
			future.whenComplete((r, t) -> {
				boolean allAvailable;
				synchronized(this) {
					synchronized(r) {
						transitionMemState(r, BlockState.WARM);
						if (r.pin() == 0)
							throw new IllegalStateException();
						_evictionCache.remove(r.getKey());
						_cache.put(r.getKey(), r);
						allAvailable = req.setPinned(idx);
					}

					if (allAvailable) {
						_processingReadRequests.remove(req);
					}

					sanityCheck();
				}
				if (allAvailable) {
					req.getFuture().complete(req.getEntries());
				}
			});
		}

		return false;
	}

	private void evict(final BlockEntry entry) {
		CompletableFuture<Void> future = _ioHandler.scheduleEviction(entry);
		future.whenComplete((r, e) -> onEvicted(entry));
	}

	private void onEvicted(final BlockEntry entry) {
		long cacheSizeDelta;
		synchronized(this) {
			synchronized(entry) {
				if(entry.isPinned()) {
					transitionMemState(entry, BlockState.WARM);
					return; // Then we cannot clear the data
				}
				cacheSizeDelta = transitionMemState(entry, BlockState.COLD);
				entry.clear();
			}
			BlockEntry tmp = _cache.remove(entry.getKey());
			if(tmp != null && tmp != entry)
				throw new IllegalStateException();
			tmp = _evictionCache.put(entry.getKey(), entry);
			if (tmp != null)
				throw new IllegalStateException();
			sanityCheck();
		}
		if (cacheSizeDelta != 0)
			onCacheSizeChanged(cacheSizeDelta > 0);
	}

	/**
	 * Cleanly transitions state of a BlockEntry and handles accounting.
	 * Requires both the scheduler object and the entry to be locked:
	 */
	private long transitionMemState(BlockEntry entry, BlockState newState) {
		BlockState oldState = entry.getState();
		if (oldState == newState)
			return 0;

		long sz = entry.getSize();
		long oldCacheSize = _cacheSize;

		// Remove old contribution
		switch (oldState) {
			case REMOVED:
				throw new IllegalStateException();
			case HOT:
			case WARM:
				_cacheSize -= sz;
				break;
			case EVICTING:
				_cacheSize -= sz;
				_bytesUpForEviction -= sz;
				break;
			case READING:
				_cacheSize -= sz;
				break;
			case COLD:
				break;
		}

		// Add new contribution
		switch (newState) {
			case REMOVED:
			case COLD:
				break;
			case HOT:
			case WARM:
				_cacheSize += sz;
				break;
			case EVICTING:
				_cacheSize += sz;
				_bytesUpForEviction += sz;
				break;
			case READING:
				_cacheSize += sz;
				break;
		}

		entry.setState(newState);
		return _cacheSize - oldCacheSize;
	}



	private static class DeferredReadRequest {
		private static final short NOT_SCHEDULED = 0;
		private static final short SCHEDULED = 1;
		private static final short PINNED = 2;

		private final CompletableFuture<List<BlockEntry>> _future;
		private final List<BlockEntry> _entries;
		private final short[] _pinned;
		private final AtomicInteger _availableCount;

		DeferredReadRequest(CompletableFuture<List<BlockEntry>> future, List<BlockEntry> entries) {
			this._future = future;
			this._entries = entries;
			this._pinned = new short[entries.size()];
			this._availableCount = new AtomicInteger(0);
		}

		CompletableFuture<List<BlockEntry>> getFuture() {
			return _future;
		}

		List<BlockEntry> getEntries() {
			return _entries;
		}

		public synchronized boolean actionRequired(int idx) {
			return _pinned[idx] == NOT_SCHEDULED;
		}

		public synchronized boolean setPinned(int idx) {
			if (_pinned[idx] == PINNED)
				return false; // already pinned
			_pinned[idx] = PINNED;
			return _availableCount.incrementAndGet() == _entries.size();
		}

		public synchronized void schedule(int idx) {
			_pinned[idx] = SCHEDULED;
		}
	}
}
