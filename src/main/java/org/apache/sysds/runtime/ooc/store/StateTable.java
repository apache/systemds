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

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.IntToLongFunction;

public final class StateTable<T extends SpillableObject> implements AutoCloseable {
	private static final int INITIAL_SLOTS = 64;

	private final OOCCache _cache;
	private final long _streamId;
	private final AtomicLong _nextGeneration = new AtomicLong();
	private final CopyOnWriteArrayList<IntToLongFunction> _evictionPolicies = new CopyOnWriteArrayList<>();
	private final AtomicBoolean _evictionPolicyInstalled = new AtomicBoolean(false);
	private Slot[] _slots;
	private volatile AtomicIntegerArray _generationSlots;
	private volatile boolean _closed;

	public StateTable(OOCCache cache, long streamId) {
		this(cache, streamId, INITIAL_SLOTS);
	}

	public StateTable(OOCCache cache, long streamId, int numSlots) {
		int capacity = Math.max(1, numSlots);
		_cache = cache;
		_streamId = streamId;
		_generationSlots = new AtomicIntegerArray(capacity);
		_slots = new Slot[capacity];
	}

	public void addEvictionPolicy(IntToLongFunction slotPolicy) {
		_evictionPolicies.add(slotPolicy);
		if(_evictionPolicyInstalled.compareAndSet(false, true))
			_cache.addEvictionPolicy(_streamId, this::scoreTableEntry);
	}

	public void put(int index, ManagedPayload<T> payload) {
		putSlot(index, slot -> finalizeOwnedPut(index, slot, payload));
	}

	public void putReference(int index, BlockEntry pinned) {
		checkPinned(pinned);
		putSlot(index, slot -> finalizeReferencePut(index, slot, pinned));
	}

	private void putSlot(int index, Consumer<Slot> finalizer) {
		Slot slot;
		synchronized(this) {
			checkOpen();
			ensureCapacity(index);
			if(_slots[index] != null)
				throw new IllegalStateException("State table slot " + index + " is already occupied.");
			slot = new Slot();
			_slots[index] = slot;
		}
		finalizer.accept(slot);
	}

	public OOCFuture<StoreLease<T>> putOrTake(int index, ManagedPayload<T> payload, MemoryAllowance leaseAllowance) {
		return putSlotOrTake(index, leaseAllowance, slot -> finalizeOwnedPut(index, slot, payload));
	}

	public OOCFuture<StoreLease<T>> putReferenceOrTake(int index, BlockEntry pinned, MemoryAllowance leaseAllowance) {
		checkPinned(pinned);
		return putSlotOrTake(index, leaseAllowance, slot -> finalizeReferencePut(index, slot, pinned));
	}

	private OOCFuture<StoreLease<T>> putSlotOrTake(int index, MemoryAllowance leaseAllowance,
		Consumer<Slot> finalizer) {
		Slot putting = null;
		Slot taken = null;
		OOCFuture<Void> waitFor = null;
		synchronized(this) {
			checkOpen();
			ensureCapacity(index);
			Slot existing = _slots[index];
			if(existing == null) {
				putting = new Slot();
				_slots[index] = putting;
			}
			else if(existing._putFuture == null) {
				_slots[index] = null;
				taken = existing;
			}
			else {
				waitFor = existing._putFuture;
			}
		}
		if(putting != null) {
			finalizer.accept(putting);
			return OOCFuture.completed(null);
		}
		if(taken != null)
			return pinTaken(taken, leaseAllowance);
		return waitFor.thenCompose(ignored -> putSlotOrTake(index, leaseAllowance, finalizer));
	}

	public OOCFuture<StoreLease<T>> take(int index, MemoryAllowance leaseAllowance) {
		Slot taken = null;
		OOCFuture<Void> waitFor = null;
		synchronized(this) {
			checkOpen();
			if(index < 0 || index >= _slots.length)
				return OOCFuture.completed(null);
			Slot existing = _slots[index];
			if(existing == null)
				return OOCFuture.completed(null);
			if(existing._putFuture == null) {
				_slots[index] = null;
				taken = existing;
			}
			else {
				waitFor = existing._putFuture;
			}
		}
		if(taken != null)
			return pinTaken(taken, leaseAllowance);
		return waitFor.thenCompose(ignored -> take(index, leaseAllowance));
	}

	public OOCFuture<StoreLease<T>> acquire(int index, MemoryAllowance leaseAllowance) {
		BlockKey key;
		synchronized(this) {
			checkOpen();
			if(index < 0 || index >= _slots.length)
				return OOCFuture.completed(null);
			Slot slot = _slots[index];
			if(slot == null || slot._putFuture != null)
				return OOCFuture.completed(null);
			key = slot._key;
		}
		OOCFuture<BlockEntry> pinned = OOCUtils.pinAdmitted(_cache, key.getStreamId(), key.getSequenceNumber(),
			leaseAllowance, () -> _closed);
		OOCFuture<StoreLease<T>> result = new OOCFuture<>();
		pinned.whenComplete((entry, error) -> {
			if(error != null)
				result.completeExceptionally(error);
			else
				result.complete(
					entry == null ? null : new StoreLease<>(entry, () -> _cache.unpin(entry, leaseAllowance)));
		});
		return result;
	}

	public StoreLease<T> peek(int index, MemoryAllowance leaseAllowance) {
		BlockKey key;
		synchronized(this) {
			checkOpen();
			if(index < 0 || index >= _slots.length)
				return null;
			Slot slot = _slots[index];
			if(slot == null || slot._putFuture != null)
				return null;
			key = slot._key;
		}
		BlockEntry entry = _cache.pinIfLive(key.getStreamId(), key.getSequenceNumber(), leaseAllowance);
		return entry == null ? null : new StoreLease<>(entry, () -> _cache.unpin(entry, leaseAllowance));
	}

	public void clear(int index) {
		Slot removed = null;
		synchronized(this) {
			if(index < 0 || index >= _slots.length)
				return;
			Slot slot = _slots[index];
			if(slot == null)
				return;
			_slots[index] = null;
			if(slot._putFuture == null)
				removed = slot;
			else
				slot._cleared = true;
		}
		if(removed != null)
			releaseSlot(removed);
	}

	@Override
	public void close() {
		List<Slot> toRelease = new ArrayList<>();
		synchronized(this) {
			if(_closed)
				return;
			_closed = true;
			for(int i = 0; i < _slots.length; i++) {
				Slot slot = _slots[i];
				if(slot == null)
					continue;
				_slots[i] = null;
				if(slot._putFuture == null)
					toRelease.add(slot);
				else
					slot._cleared = true;
			}
		}
		for(Slot slot : toRelease)
			releaseSlot(slot);
	}

	private void finalizeOwnedPut(int index, Slot slot, ManagedPayload<T> payload) {
		BlockKey key = new BlockKey(_streamId, _nextGeneration.getAndIncrement());
		BlockEntry entry;
		try {
			payload.transfer();
		}
		catch(RuntimeException ex) {
			failPut(index, slot, ex);
			throw ex;
		}
		try {
			entry = _cache.putPinned(key.getStreamId(), key.getSequenceNumber(), payload.value(), payload.bytes(),
				payload.owner());
		}
		catch(RuntimeException ex) {
			if(payload.bytes() > 0)
				payload.owner().release(payload.bytes());
			failPut(index, slot, ex);
			throw ex;
		}
		boolean cleared;
		OOCFuture<Void> putFuture;
		synchronized(this) {
			slot._key = key;
			slot._tableOwnedKey = true;
			int generation = blockIndex(key.getSequenceNumber());
			ensureGenerationCapacity(generation);
			_generationSlots.set(generation, index + 1);
			cleared = slot._cleared;
			putFuture = slot._putFuture;
			slot._putFuture = null;
		}
		_cache.unpin(entry, payload.owner());
		if(cleared)
			releaseSlot(slot);
		putFuture.complete(null);
	}

	private void finalizeReferencePut(int index, Slot slot, BlockEntry pinned) {
		try {
			_cache.reference(pinned);
		}
		catch(RuntimeException ex) {
			failPut(index, slot, ex);
			throw ex;
		}

		boolean cleared;
		OOCFuture<Void> putFuture;
		synchronized(this) {
			slot._key = pinned.getKey();
			slot._tableOwnedKey = false;
			cleared = slot._cleared;
			putFuture = slot._putFuture;
			slot._putFuture = null;
		}
		if(cleared)
			_cache.dereference(pinned.getKey());
		putFuture.complete(null);
	}

	private void failPut(int index, Slot slot, RuntimeException ex) {
		OOCFuture<Void> putFuture;
		synchronized(this) {
			if(index < _slots.length && _slots[index] == slot)
				_slots[index] = null;
			putFuture = slot._putFuture;
			slot._putFuture = null;
		}
		if(putFuture != null)
			putFuture.completeExceptionally(ex);
	}

	private OOCFuture<StoreLease<T>> pinTaken(Slot slot, MemoryAllowance leaseAllowance) {
		OOCFuture<BlockEntry> pinned = OOCUtils.pinAdmitted(_cache, slot._key.getStreamId(),
			slot._key.getSequenceNumber(), leaseAllowance, () -> _closed);
		OOCFuture<StoreLease<T>> result = new OOCFuture<>();
		pinned.whenComplete((entry, error) -> {
			Throwable completionError = error;
			try {
				releaseSlot(slot);
			}
			catch(RuntimeException releaseError) {
				if(completionError == null)
					completionError = releaseError;
			}
			if(completionError == null && entry == null)
				completionError = new IllegalStateException("State table closed while a take was pending.");
			if(completionError != null) {
				if(entry != null) {
					try {
						_cache.unpin(entry, leaseAllowance);
					}
					catch(RuntimeException ignored) {
					}
				}
				result.completeExceptionally(completionError);
				return;
			}
			result.complete(new StoreLease<>(entry, () -> _cache.unpin(entry, leaseAllowance)));
		});
		return result;
	}

	private void releaseSlot(Slot slot) {
		if(slot._tableOwnedKey) {
			int generation = blockIndex(slot._key.getSequenceNumber());
			AtomicIntegerArray slots = _generationSlots;
			if(generation < slots.length())
				slots.set(generation, 0);
		}
		_cache.dereference(slot._key);
	}

	private long scoreTableEntry(long generation) {
		int index = blockIndex(generation);
		AtomicIntegerArray slots = _generationSlots;
		if(index >= slots.length())
			return Long.MAX_VALUE;
		int encodedSlot = slots.get(index);
		if(encodedSlot == 0)
			return Long.MAX_VALUE;
		int slot = encodedSlot - 1;
		long score = Long.MAX_VALUE;
		for(IntToLongFunction policy : _evictionPolicies)
			score = Math.min(score, policy.applyAsLong(slot));
		return score;
	}

	private void ensureGenerationCapacity(int index) {
		AtomicIntegerArray slots = _generationSlots;
		if(index < slots.length())
			return;
		int newLength = slots.length();
		while(index >= newLength) {
			if(newLength > Integer.MAX_VALUE / 2)
				throw new IllegalStateException("State table generation map capacity overflow");
			newLength *= 2;
		}
		AtomicIntegerArray grown = new AtomicIntegerArray(newLength);
		for(int i = 0; i < slots.length(); i++)
			grown.set(i, slots.get(i));
		_generationSlots = grown;
	}

	private static int blockIndex(long sequenceNumber) {
		if(sequenceNumber < 0 || sequenceNumber > Integer.MAX_VALUE)
			throw new IndexOutOfBoundsException("Invalid block index: " + sequenceNumber);
		return (int) sequenceNumber;
	}

	private void checkOpen() {
		if(_closed)
			throw new IllegalStateException("State table is closed.");
	}

	private static void checkPinned(BlockEntry pinned) {
		if(!pinned.isPinned())
			throw new IllegalArgumentException(
				"Reference install requires the supplied entry to be pinned: " + pinned.getKey());
	}

	private void ensureCapacity(int index) {
		if(index < 0)
			throw new IndexOutOfBoundsException("Invalid slot index: " + index);
		if(index < _slots.length)
			return;
		int newLength = _slots.length;
		while(index >= newLength)
			newLength *= 2;
		Slot[] grown = new Slot[newLength];
		System.arraycopy(_slots, 0, grown, 0, _slots.length);
		_slots = grown;
	}

	private static final class Slot {
		private boolean _cleared;
		private boolean _tableOwnedKey;
		private BlockKey _key;
		private OOCFuture<Void> _putFuture = new OOCFuture<>();
	}
}
