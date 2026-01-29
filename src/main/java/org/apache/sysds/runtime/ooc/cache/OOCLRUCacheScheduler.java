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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.utils.Statistics;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class OOCLRUCacheScheduler implements OOCCacheScheduler {
	private static final boolean SANITY_CHECKS = false;
	private static final Log LOG = LogFactory.getLog(OOCLRUCacheScheduler.class.getName());
	private final OOCIOHandler _ioHandler;
	private final LinkedHashMap<BlockKey, BlockEntry> _cache;
	private final HashMap<BlockKey, BlockEntry> _evictionCache;
	private final DeferredReadQueue _deferredReadRequests;
	private final Deque<DeferredReadRequest> _processingReadRequests;
	private final HashMap<BlockKey, BlockReadState> _blockReads;
	private long _hardLimit;
	private long _evictionLimit;
	private final int _callerId;
	private long _cacheSize;
	private long _bytesUpForEviction;
	private volatile boolean _running;
	private boolean _warnThrottling;

	public OOCLRUCacheScheduler(OOCIOHandler ioHandler, long evictionLimit, long hardLimit) {
		this._ioHandler = ioHandler;
		this._cache = new LinkedHashMap<>(1024, 0.75f, true);
		this._evictionCache = new  HashMap<>();
		this._deferredReadRequests = new DeferredReadQueue();
		this._processingReadRequests = new ArrayDeque<>();
		this._blockReads = new HashMap<>();
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

		// Schedule deferred read otherwise
		final CompletableFuture<List<BlockEntry>> requestFuture = new CompletableFuture<>();
		CompletableFuture<BlockEntry> future = requestFuture.thenApply(l -> l.get(0));
		scheduleDeferredRead(new DeferredReadRequest(requestFuture, Collections.singletonList(entry)));
		return future;
	}

	@Override
	public List<BlockEntry> tryRequest(List<BlockKey> keys) {
		CompletableFuture<List<BlockEntry>> f = request(keys, true);
		if(f == null)
			return null;
		return f.getNow(null);
	}

	@Override
	public CompletableFuture<List<BlockEntry>> requestAnyOf(List<BlockKey> keys, int n, List<BlockKey> selectionOut) {
		List<BlockEntry> l = tryRequestAnyOf(keys, n, selectionOut);
		if(l != null)
			return CompletableFuture.completedFuture(l);
		return request(keys.subList(0, n));
	}

	@Override
	public List<BlockEntry> tryRequestAnyOf(List<BlockKey> keys, int n, List<BlockKey> selectionOut) {
		List<BlockEntry> present = new ArrayList<>(n);
		for(BlockKey key : keys) {
			List<BlockEntry> l = tryRequest(List.of(key));
			if(l != null) {
				present.add(l.get(0));
				selectionOut.add(l.get(0).getKey());
				if(l.size() == n)
					return l;
			}
		}
		present.forEach(this::unpin);
		return null;
	}

	@Override
	public CompletableFuture<List<BlockEntry>> request(List<BlockKey> keys) {
		return request(keys, false);
	}

	public CompletableFuture<List<BlockEntry>> request(List<BlockKey> keys, boolean onlyIfAvailable) {
		if (!this._running)
			throw new IllegalStateException("Cache scheduler has been shut down.");

		Statistics.incrementOOCEvictionGet(keys.size());

		List<BlockEntry> entries = new ArrayList<>(keys.size());
		boolean allAvailable = true;

		synchronized(this) {
			for (BlockKey key : keys) {
				BlockEntry entry = _cache.get(key);
				if (entry == null)
					entry = _evictionCache.get(key);
				if (entry == null)
					throw new IllegalArgumentException("Could not find requested block with key " + key);

				synchronized(entry) {
					if(!entry.getState().isAvailable())
						allAvailable = false;
				}
				entries.add(entry);
			}

			if(allAvailable) {
				for(BlockEntry entry : entries) {
					synchronized(entry) {
						if(entry.pin() == 0)
							throw new IllegalStateException();
					}
				}
			}
		}

		if (allAvailable) {
			// Then we could pin all entries
			return CompletableFuture.completedFuture(entries);
		}

		if(onlyIfAvailable)
			return null;

		// Schedule deferred read otherwise
		final  CompletableFuture<List<BlockEntry>> future = new CompletableFuture<>();
		DeferredReadRequest request = new DeferredReadRequest(future, entries);
		for (int i = 0; i < entries.size(); i++) {
			BlockEntry entry = entries.get(i);
			synchronized(entry) {
				if (entry.getState().isAvailable()) {
					entry.addRetainHint();
					request.markRetainHinted(i);
				}
			}
		}
		scheduleDeferredRead(request);
		return future;
	}

	@Override
	public void prioritize(BlockKey key, double priority) {
		if (!this._running)
			return;
		if (priority == 0)
			return;

		synchronized(this) {
			boolean matched = _deferredReadRequests.boost(key, priority);
			if(matched) {
				BlockReadState state = _blockReads.computeIfAbsent(key, k -> new BlockReadState());
				state.priority += priority;
			}
		}
		_ioHandler.prioritizeRead(key, priority);
	}

	private void scheduleDeferredRead(DeferredReadRequest deferredReadRequest) {
		synchronized(this) {
			double score = 0;
			int readyCount = 0;
			for (BlockEntry entry : deferredReadRequest.getEntries()) {
				synchronized(entry) {
					if (entry.getState().isAvailable())
						readyCount++;
				}
				BlockReadState state = _blockReads.get(entry.getKey());
				if (state != null)
					score += state.priority;
			}
			if (!deferredReadRequest.getEntries().isEmpty())
				score /= deferredReadRequest.getEntries().size();
			if (!deferredReadRequest.getEntries().isEmpty())
				score += ((double) readyCount) / deferredReadRequest.getEntries().size();
			deferredReadRequest.setPriorityScore(score);

			_deferredReadRequests.add(deferredReadRequest);
		}
		onCacheSizeChanged(false); // To schedule deferred reads if possible
	}

	@Override
	public BlockKey put(BlockKey key, Object data, long size) {
		return put(key, data, size, false, null).getKey();
	}

	@Override
	public BlockEntry putAndPin(BlockKey key, Object data, long size) {
		return put(key, data, size, true, null);
	}

	@Override
	public void putSourceBacked(BlockKey key, Object data, long size, OOCIOHandler.SourceBlockDescriptor descriptor) {
		put(key, data, size, false, descriptor);
	}

	@Override
	public BlockEntry putAndPinSourceBacked(BlockKey key, Object data, long size, OOCIOHandler.SourceBlockDescriptor descriptor) {
		return put(key, data, size, true, descriptor);
	}

	private BlockEntry put(BlockKey key, Object data, long size, boolean pin, OOCIOHandler.SourceBlockDescriptor descriptor) {
		if (!this._running)
			throw new IllegalStateException();
		if (data == null)
			throw new IllegalArgumentException();
		if (descriptor != null)
			_ioHandler.registerSourceLocation(key, descriptor);

		Statistics.incrementOOCEvictionPut();
		BlockEntry entry = new BlockEntry(key, size, data);
		if (descriptor != null)
			entry.setState(BlockState.WARM);
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
					entry.setDataUnsafe(null);
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
			boolean shouldCheckEviction = false;
			synchronized(this) {
				if (_cacheSize <= _evictionLimit)
					return; // Nothing to do

				synchronized(entry) {
					if (entry.isPinned())
						return; // Pin state changed so we cannot evict

					if (entry.getState().isAvailable() && entry.getState().isBackedByDisk()) {
						if (entry.getRetainHintCount() > 0) {
							shouldCheckEviction = true;
						}
						else {
							cacheSizeDelta =  transitionMemState(entry, BlockState.COLD);
							long cleared = entry.clear();
							if (cleared != entry.getSize())
								throw new IllegalStateException();
							_cache.remove(entry.getKey());
							_evictionCache.put(entry.getKey(), entry);
						}
					} else if (entry.getState() == BlockState.HOT) {
						if (entry.getRetainHintCount() > 0) {
							shouldCheckEviction = true;
						}
						else {
							cacheSizeDelta = onUnpinnedHotBlockUnderMemoryPressure(entry);
						}
					}
				}
			}
			if (cacheSizeDelta != 0)
				onCacheSizeChanged(cacheSizeDelta > 0);
			else if (shouldCheckEviction)
				onCacheSizeChanged(true);
		}
	}

	@Override
	public synchronized long getCacheSize() {
		return _cacheSize;
	}

	@Override
	public boolean isWithinLimits() {
		return _cacheSize < _hardLimit;
	}

	@Override
	public boolean isWithinSoftLimits() {
		return _cacheSize < _evictionLimit;
	}

	@Override
	public synchronized void shutdown() {
		this._running = false;
		_cache.clear();
		_evictionCache.clear();
		_processingReadRequests.clear();
		_deferredReadRequests.clear();
		_blockReads.clear();
		_cacheSize = 0;
		_bytesUpForEviction = 0;
	}

	@Override
	public synchronized void updateLimits(long evictionLimit, long hardLimit) {
		_evictionLimit = evictionLimit;
		_hardLimit = hardLimit;
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
		if (_cacheSize > _hardLimit * 1.1) {
			if (!_warnThrottling) {
				_warnThrottling = true;
				System.out.println("[WARN] Cache hard limit exceeded by over 10%: " + String.format("%.2f", _cacheSize/1000000.0) + "MB (-" + String.format("%.2f", _bytesUpForEviction/1000000.0) + "MB) > " + String.format("%.2f", _hardLimit/1000000.0) + "MB");
			}
		}
		else if (_warnThrottling && _cacheSize < _hardLimit) {
			_warnThrottling = false;
			System.out.println("[INFO] Cache within limit: " + String.format("%.2f", _cacheSize/1000000.0) + "MB (-" + String.format("%.2f", _bytesUpForEviction/1000000.0) + "MB) <= " + String.format("%.2f", _hardLimit/1000000.0) + "MB");
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

			for(int pass = 0; pass < 2; pass++) {
				boolean allowRetainHint = pass == 1;
				for(BlockEntry entry : entries) {
					if(_cacheSize - _bytesUpForEviction <= _evictionLimit)
						break;

					synchronized(entry) {
						if(entry.isPinned())
							continue;
						if(!allowRetainHint && entry.getRetainHintCount() > 0)
							continue;
						if(entry.getState() == BlockState.COLD || entry.getState() == BlockState.EVICTING)
							continue;

						if(entry.getState().isBackedByDisk()) {
							cacheSizeDelta += transitionMemState(entry, BlockState.COLD);
							entry.clear();
							toRemove.add(entry);
						}
						else {
							cacheSizeDelta += transitionMemState(entry, BlockState.EVICTING);
							upForEviction.add(entry);
						}
					}
				}
				if(_cacheSize - _bytesUpForEviction <= _evictionLimit)
					break;
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
		boolean reading = false;
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
					else if (entry.getState() == BlockState.READING) {
						req.schedule(idx);
						registerWaiter(entry.getKey(), req, idx);
						reading = true;
					}
					else {
						if(_cacheSize + entry.getSize() <= _hardLimit) {
							transitionMemState(entry, BlockState.READING);
							toRead.add(new Tuple2<>(idx, entry));
							req.schedule(idx);
							registerWaiter(entry.getKey(), req, idx);
							reading = true;
						}
						else {
							allReserved = false;
						}
					}
				}
			}

			if(allReserved) {
				_deferredReadRequests.poll();
				if (!toRead.isEmpty())
					_processingReadRequests.add(req);
			}

			sanityCheck();
		}

		if(allReserved && !reading) {
			clearRetainHints(req);
			req.getFuture().complete(req.getEntries());
			return true;
		}
		else if(allReserved && reading && req.isComplete()) {
			clearRetainHints(req);
			synchronized(this) {
				_processingReadRequests.remove(req);
				_deferredReadRequests.remove(req);
			}
			req.getFuture().complete(req.getEntries());
			return true;
		}

		for(Tuple2<Integer, BlockEntry> tpl : toRead) {
			final BlockEntry entry = tpl._2;
			CompletableFuture<BlockEntry> future = _ioHandler.scheduleRead(entry);
			future.whenComplete((r, t) -> {
				if(t != null) {
					BlockReadState state;
					synchronized(OOCLRUCacheScheduler.this) {
						state = _blockReads.remove(entry.getKey());

					}
					if(state != null) {
						for(DeferredReadWaiter waiter : state.waiters)
							waiter.request.getFuture().completeExceptionally(t);
					}
					else {
						LOG.error("Uncaught CacheError", t);
						t.printStackTrace();
					}
					return;
				}
				java.util.Set<DeferredReadRequest> completedRequests = new java.util.HashSet<>();
				synchronized(this) {
					synchronized(r) {
						transitionMemState(r, BlockState.WARM);
						_evictionCache.remove(r.getKey());
						_cache.put(r.getKey(), r);
					}

					BlockReadState state = _blockReads.remove(r.getKey());
					if(state != null) {
						for(DeferredReadWaiter waiter : state.waiters) {
							synchronized(r) {
								if(r.pin() == 0)
									throw new IllegalStateException();
								if(waiter.request.setPinned(waiter.index) || waiter.request.isComplete())
									completedRequests.add(waiter.request);
							}
						}
					}

					for(DeferredReadRequest done : completedRequests) {
						clearRetainHints(done);
						_processingReadRequests.remove(done);
						_deferredReadRequests.remove(done);
					}

					sanityCheck();
				}
				for(DeferredReadRequest done : completedRequests)
					done.getFuture().complete(done.getEntries());
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
				if(entry.getState() == BlockState.REMOVED)
					return;
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

	private void clearRetainHints(DeferredReadRequest request) {
		for (int i = 0; i < request.getEntries().size(); i++) {
			if (!request.isRetainHinted(i))
				continue;
			BlockEntry entry = request.getEntries().get(i);
			synchronized(entry) {
				entry.removeRetainHint();
			}
		}
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

	private void registerWaiter(BlockKey key, DeferredReadRequest request, int index) {
		BlockReadState state = _blockReads.computeIfAbsent(key, k -> new BlockReadState());
		state.waiters.add(new DeferredReadWaiter(request, index));
	}

	private static class BlockReadState {
		private double priority;
		private final List<DeferredReadWaiter> waiters;

		private BlockReadState() {
			this.priority = 0;
			this.waiters = new ArrayList<>();
		}
	}

	private static class DeferredReadWaiter {
		private final DeferredReadRequest request;
		private final int index;

		private DeferredReadWaiter(DeferredReadRequest request, int index) {
			this.request = request;
			this.index = index;
		}
	}
}
