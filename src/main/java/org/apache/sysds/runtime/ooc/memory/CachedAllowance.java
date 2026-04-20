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
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCCacheScheduler;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class CachedAllowance extends SyncMemoryAllowance {
	private static final int INITIAL_SLOTS = 64;
	private static final long MIN_HANDOVER_SLACK = 1_000_000L;
	private static final long MAX_HANDOVER_SLACK = 128_000_000L;

	private final long _streamId;
	private final AtomicLong _nextBlockId;
	private volatile AtomicReferenceArray<SlotEntry> _slots;
	private long _pendingHandoverBytes;
	private int _highestPopulatedIndex;
	private boolean _handoverScheduling;
	private boolean _handoverSchedulingRequested;

	public CachedAllowance(MemoryBroker broker) {
		super(broker);
		_streamId = CachingStream._streamSeq.getNextID();
		_slots = new AtomicReferenceArray<>(INITIAL_SLOTS);
		_nextBlockId = new AtomicLong(0);
		_pendingHandoverBytes = 0;
		_highestPopulatedIndex = -1;
		_handoverScheduling = false;
		_handoverSchedulingRequested = false;
	}

	public void handover(InMemoryQueueCallback callback, int index) {
		if(callback == null)
			throw new IllegalArgumentException("Cannot hand over null callback.");
		callback.transferOwnershipBlocking(this);

		InMemoryQueueCallback root = (InMemoryQueueCallback) callback.keepOpen();
		callback.close();
		root.getHandle().attachCachedAllowance(this, index);

		SlotEntry entry = new SlotEntry(root);
		synchronized(this) {
			ensureCapacity(index);
			AtomicReferenceArray<SlotEntry> slots = _slots;
			if(slots.get(index) != null) {
				root.getHandle().detachCachedAllowance();
				root.close();
				throw new IllegalStateException("Cached allowance slot " + index + " already occupied.");
			}
			slots.set(index, entry);
			if(index > _highestPopulatedIndex)
				_highestPopulatedIndex = index;
		}
	}

	public OOCStream.QueueCallback<IndexedMatrixValue> tryGet(int index) {
		SlotEntry entry = getSlot(index);
		if(entry == null)
			return null;

		while(true) {
			BlockKey cacheKey = null;
			OOCCacheScheduler.HandoverHandle handover = null;
			InMemoryQueueCallback local = null;

			synchronized(entry) {
				if(entry._local != null && entry._handover == null)
					local = entry._local;
				else if(entry._handover != null) {
					handover = entry._handover;
					cacheKey = entry._cacheKey;
				}
				else if(entry._cacheKey != null)
					cacheKey = entry._cacheKey;
				else
					return null;
			}

			if(local != null)
				return local.keepOpen();

			if(handover != null) {
				OOCStream.QueueCallback<IndexedMatrixValue> reclaimed = handover.reclaim();
				if(reclaimed != null) {
					reclaimed.close();
					synchronized(entry) {
						if(entry._handover == handover) {
							finishPendingHandover(entry);
							entry._handover = null;
							entry._cacheKey = null;
						}
					}
					continue;
				}

				CompletableFuture<Boolean> future = handover.getCompletionFuture();
				if(!future.isDone())
					return null;
				boolean committed = future.join();
				InMemoryQueueCallback localToClose = null;
				synchronized(entry) {
					if(entry._handover != handover)
						continue;
					finishPendingHandover(entry);
					entry._handover = null;
					if(committed) {
						localToClose = entry._local;
						entry._local = null;
					}
					else {
						entry._cacheKey = null;
					}
				}
				if(localToClose != null)
					closeRoot(localToClose);
				continue;
			}

			return OOCCacheManager.tryRequestBlock(cacheKey);
		}
	}

	public CompletableFuture<OOCStream.QueueCallback<IndexedMatrixValue>> get(int index) {
		OOCStream.QueueCallback<IndexedMatrixValue> immediate = tryGet(index);
		if(immediate != null)
			return CompletableFuture.completedFuture(immediate);

		SlotEntry entry = getSlot(index);
		if(entry == null)
			return CompletableFuture.completedFuture(null);

		OOCCacheScheduler.HandoverHandle handover;
		BlockKey cacheKey;
		synchronized(entry) {
			if(entry._local != null && entry._handover == null)
				return CompletableFuture.completedFuture(entry._local.keepOpen());
			handover = entry._handover;
			cacheKey = entry._cacheKey;
		}

		if(handover != null) {
			return handover.getCompletionFuture().handle((committed, ex) -> {
				if(ex != null)
					throw DMLRuntimeException.of(ex.getCause() == null ? ex : ex.getCause());
				return committed == true;
			}).thenCompose(committed -> {
				InMemoryQueueCallback localToClose = null;
				InMemoryQueueCallback local = null;
				BlockKey key;

				synchronized(entry) {
					if(entry._handover != handover)
						return get(index);

					finishPendingHandover(entry);
					entry._handover = null;
					if(committed) {
						key = entry._cacheKey;
						localToClose = entry._local;
						entry._local = null;
					}
					else {
						entry._cacheKey = null;
						local = entry._local;
						key = null;
					}
				}

				if(localToClose != null)
					closeRoot(localToClose);

				if(committed)
					return OOCCacheManager.requestBlock(key);
				return CompletableFuture.completedFuture(local == null ? null : local.keepOpen());
			});
		}

		if(cacheKey != null)
			return OOCCacheManager.requestBlock(cacheKey);
		return CompletableFuture.completedFuture(null);
	}

	public void clear(int index) {
		SlotEntry entry = removeSlot(index);
		if(entry == null)
			return;

		while(true) {
			OOCCacheScheduler.HandoverHandle handover = null;
			BlockKey forgetKey = null;
			InMemoryQueueCallback localToClose = null;

			synchronized(entry) {
				if(entry._local != null && entry._handover == null) {
					localToClose = entry._local;
					entry._local = null;
				}
				else if(entry._handover != null)
					handover = entry._handover;
				else if(entry._cacheKey != null) {
					forgetKey = entry._cacheKey;
					entry._cacheKey = null;
				}
				else
					return;
			}

			if(localToClose != null) {
				closeRoot(localToClose);
				return;
			}

			if(forgetKey != null) {
				OOCCacheManager.forget(forgetKey);
				return;
			}

			OOCStream.QueueCallback<IndexedMatrixValue> reclaimed = handover.reclaim();
			if(reclaimed != null) {
				reclaimed.close();
				synchronized(entry) {
					if(entry._handover == handover) {
						finishPendingHandover(entry);
						localToClose = entry._local;
						entry._local = null;
						entry._handover = null;
						entry._cacheKey = null;
					}
				}
				if(localToClose != null)
					closeRoot(localToClose);
				return;
			}

			boolean committed;
			try {
				committed = handover.getCompletionFuture().join();
			}
			catch(CompletionException ex) {
				throw DMLRuntimeException.of(ex.getCause() == null ? ex : ex.getCause());
			}

			synchronized(entry) {
				if(entry._handover != handover)
					continue;
				finishPendingHandover(entry);
				localToClose = entry._local;
				entry._local = null;
				entry._handover = null;
				if(committed)
					forgetKey = entry._cacheKey;
				entry._cacheKey = null;
			}

			if(localToClose != null)
				closeRoot(localToClose);
			if(forgetKey != null)
				OOCCacheManager.forget(forgetKey);
			return;
		}
	}

	@Override
	public boolean tryReserve(long bytes) {
		throw new UnsupportedOperationException("CachedAllowance does not support direct reservations. Use handover(...).");
	}

	@Override
	public void reserveBlocking(long bytes) {
		throw new UnsupportedOperationException("CachedAllowance does not support direct reservations. Use handover(...).");
	}

	@Override
	public void setTargetMemory(long targetMemory) {
		super.setTargetMemory(targetMemory);
		maybeScheduleHandovers(0);
	}

	void onFinishedHandover(long bytes) {
		synchronized(this) {
			_pendingHandoverBytes -= bytes;
			if(_pendingHandoverBytes < 0)
				throw new IllegalStateException();
			notifyAll();
		}
		maybeScheduleHandovers(0);
	}

	void admitBlocking(long bytes) {
		while(true) {
			if(super.tryReserve(bytes))
				return;
			maybeScheduleHandovers(bytes);
			if(super.tryReserve(bytes))
				return;
			if(_shutdown || _destroyed)
				throw new IllegalStateException("Cannot reserve memory on closed allowance.");
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

	private void maybeScheduleHandovers(long requestedBytes) {
		synchronized(this) {
			_handoverSchedulingRequested = true;
			if(_handoverScheduling)
				return;
			_handoverScheduling = true;
		}

		boolean restart;
		try {
			while(true) {
				long reclaimGoal;
				int startIndex;
				synchronized(this) {
					_handoverSchedulingRequested = false;
					if(_shutdown || _destroyed)
						return;
					long excess = _usedBytes + requestedBytes - _targetBytes - _pendingHandoverBytes;
					if(excess <= 0) {
						if(!_handoverSchedulingRequested)
							return;
						continue;
					}

					long slack = Math.max(MIN_HANDOVER_SLACK, Math.min(MAX_HANDOVER_SLACK, _targetBytes / 16));
					reclaimGoal = excess + slack;
					startIndex = _highestPopulatedIndex;
				}

				AtomicReferenceArray<SlotEntry> slots = _slots;
				int newHighest = startIndex;
				// Find highes non-null entry
				for(int i = Math.min(startIndex, slots.length() - 1); i >= 0; i--) {
					if(slots.get(i) != null) {
						newHighest = i;
						break;
					}
				}

				for(int i = newHighest; i >= 0 && reclaimGoal > 0; i--) {
					long bytes = tryStartCacheHandover(slots.get(i));
					if(bytes <= 0)
						continue;
					reclaimGoal -= bytes;
				}

				synchronized(this) {
					if(newHighest < _highestPopulatedIndex)
						_highestPopulatedIndex = newHighest;
					if(!_handoverSchedulingRequested)
						return;
				}
			}
		}
		finally {
			synchronized(this) {
				_handoverScheduling = false;
				restart = _handoverSchedulingRequested;
			}
			if(restart)
				maybeScheduleHandovers(requestedBytes);
		}
	}

	private long tryStartCacheHandover(SlotEntry entry) {
		if(entry == null)
			return 0;
		synchronized(entry) {
			if(entry._local == null || entry._handover != null || !entry._local.getHandle().isExclusiveToRoot())
				return 0;

			long bytes = entry._local.getManagedBytes();
			if(bytes <= 0)
				return 0;

			InMemoryQueueCallback retained = (InMemoryQueueCallback) entry._local.keepOpen();
				try {
					entry._cacheKey = new BlockKey(_streamId, _nextBlockId.getAndIncrement());
					entry._handover = OOCCacheManager.handover(entry._cacheKey, retained);
					entry._pendingBytes = bytes;
					synchronized(this) {
						_pendingHandoverBytes += bytes;
					}
					entry._handover.getCompletionFuture().whenComplete((committed, ex) -> onHandoverCompleted(entry));
					return bytes;
				}
			catch(RuntimeException ex) {
				entry._cacheKey = null;
				entry._handover = null;
				entry._pendingBytes = 0;
				retained.close();
				throw ex;
			}
		}
	}

	private void onHandoverCompleted(SlotEntry entry) {
		synchronized(entry) {
			if(entry._pendingBytes <= 0)
				return;
			finishPendingHandover(entry);
		}
		maybeScheduleHandovers(0);
	}

	private void finishPendingHandover(SlotEntry entry) {
		if(entry._pendingBytes <= 0)
			return;
		long bytes = entry._pendingBytes;
		entry._pendingBytes = 0;
		onFinishedHandover(bytes);
	}

	private void closeRoot(InMemoryQueueCallback local) {
		local.getHandle().detachCachedAllowance();
		local.close();
	}

	private SlotEntry getSlot(int index) {
		AtomicReferenceArray<SlotEntry> slots = _slots;
		if(index < 0 || index >= slots.length())
			return null;
		return slots.get(index);
	}

	private SlotEntry removeSlot(int index) {
		synchronized(this) {
			AtomicReferenceArray<SlotEntry> slots = _slots;
			if(index < 0 || index >= slots.length())
				return null;
			SlotEntry entry = slots.get(index);
			if(entry != null)
				slots.set(index, null);
			return entry;
		}
	}

	private void ensureCapacity(int index) {
		AtomicReferenceArray<SlotEntry> slots = _slots;
		if(index < slots.length())
			return;
		int newLen = slots.length();
		while(index >= newLen)
			newLen *= 2;
		AtomicReferenceArray<SlotEntry> grown = new AtomicReferenceArray<>(newLen);
		for(int i = 0; i < slots.length(); i++)
			grown.set(i, slots.get(i));
		_slots = grown;
	}

	private static final class SlotEntry {
		private InMemoryQueueCallback _local;
		private BlockKey _cacheKey;
		private OOCCacheScheduler.HandoverHandle _handover;
		private long _pendingBytes;

		private SlotEntry(InMemoryQueueCallback local) {
			_local = local;
		}
	}
}
