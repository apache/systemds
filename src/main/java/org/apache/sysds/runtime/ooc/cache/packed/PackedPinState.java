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

package org.apache.sysds.runtime.ooc.cache.packed;

import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

final class PackedPinState {
	final BlockEntry physicalEntry;
	final OOCPackedCache.PackGroup group;
	private MemoryAllowance[] _allowances;
	private int[] _counts;
	private OOCFuture<BlockEntry>[] _futures;
	private long[] _releaseDueNanos;
	private PackedUnpinHandle[] _releaseHandles;
	private int _size;
	private boolean _releaseQueued;
	private int _liveLocations;

	@SuppressWarnings("unchecked")
	PackedPinState(BlockEntry physicalEntry, long streamId, int[] tileIds, int off, int count, int liveLocations) {
		this.physicalEntry = physicalEntry;
		group = new OOCPackedCache.PackGroup(this, streamId, tileIds, off, count);
		this._liveLocations = liveLocations;
		_allowances = new MemoryAllowance[2];
		_counts = new int[2];
		_futures = new OOCFuture[2];
		_releaseDueNanos = new long[2];
		_releaseHandles = new PackedUnpinHandle[2];
	}

	synchronized OOCFuture<BlockEntry> pin(OOCCacheImpl physical, MemoryAllowance allowance, boolean liveOnly) {
		int ix = indexOf(allowance);
		if(ix >= 0) {
			cancelRelease(ix);
			_counts[ix]++;
			return _futures[ix];
		}
		OOCFuture<BlockEntry> future = liveOnly ? OOCFuture.completed(
			physical.pinIfLive(physicalEntry.getKey().getStreamId(), physicalEntry.getKey().getSequenceNumber(),
				allowance)) : physical.pin(physicalEntry.getKey(), allowance);
		addAllowance(allowance, future);
		future.whenComplete((entry, ex) -> {
			if(entry == null || ex != null)
				removeFailedAllowance(allowance, future);
		});
		return future;
	}

	synchronized OOCFuture<BlockEntry> pinAdmitted(OOCCacheImpl physical, MemoryAllowance allowance) {
		int ix = indexOf(allowance);
		if(ix >= 0) {
			cancelRelease(ix);
			_counts[ix]++;
			return _futures[ix];
		}
		OOCFuture<BlockEntry> future = physical.pinAdmitted(physicalEntry.getKey(), allowance);
		addAllowance(allowance, future);
		future.whenComplete((entry, ex) -> {
			if(entry == null || ex != null)
				removeFailedAllowance(allowance, future);
		});
		return future;
	}

	BlockEntry pinIfLive(OOCCacheImpl physical, MemoryAllowance allowance) {
		try {
			return pin(physical, allowance, true).getNow(null);
		}
		catch(RuntimeException ex) {
			return null;
		}
	}

	synchronized OOCCache.UnpinHandle unpin(OOCPackedCache owner, long releaseDelayMs, MemoryAllowance allowance) {
		int ix = indexOf(allowance);
		if(ix < 0)
			return PackedUnpinHandle.committed(physicalEntry, allowance, physicalEntry.getSize());
		_counts[ix]--;
		if(_counts[ix] > 0)
			return PackedUnpinHandle.committed(physicalEntry, allowance, physicalEntry.getSize());
		PackedUnpinHandle handle = PackedUnpinHandle.delayedPhysicalRelease(physicalEntry, allowance);
		_releaseHandles[ix] = handle;
		_releaseDueNanos[ix] = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(0, releaseDelayMs));
		owner.enqueueRelease(this);
		return handle;
	}

	long releaseDuePins(OOCCacheImpl physical, long nowNanos) {
		ArrayList<PackedRelease> due = null;
		long nextDueNanos = Long.MAX_VALUE;
		synchronized(this) {
			for(int i = 0; i < _size;) {
				PackedUnpinHandle handle = _releaseHandles[i];
				if(handle == null || _counts[i] > 0) {
					i++;
					continue;
				}
				long dueNanos = _releaseDueNanos[i];
				if(dueNanos > nowNanos) {
					nextDueNanos = Math.min(nextDueNanos, dueNanos);
					i++;
					continue;
				}
				if(due == null)
					due = new ArrayList<>();
				due.add(new PackedRelease(_allowances[i], handle));
				removeAt(i);
			}
		}
		if(due != null)
			for(PackedRelease release : due)
				releasePhysicalPin(physical, release.allowance, release.handle);
		return nextDueNanos;
	}

	synchronized boolean markReleaseQueued() {
		if(_releaseQueued)
			return false;
		_releaseQueued = true;
		return true;
	}

	synchronized void clearReleaseQueued() {
		_releaseQueued = false;
	}

	synchronized boolean forgetLocation() {
		if(_liveLocations <= 0)
			return false;
		return --_liveLocations == 0;
	}

	private int indexOf(MemoryAllowance allowance) {
		for(int i = 0; i < _size; i++)
			if(_allowances[i] == allowance)
				return i;
		return -1;
	}

	private synchronized void addAllowance(MemoryAllowance allowance, OOCFuture<BlockEntry> future) {
		if(_size == _allowances.length)
			grow();
		_allowances[_size] = allowance;
		_counts[_size] = 1;
		_futures[_size] = future;
		_size++;
	}

	@SuppressWarnings("unchecked")
	private void grow() {
		int nextSize = _size * 2;
		MemoryAllowance[] biggerAllowances = new MemoryAllowance[nextSize];
		int[] biggerCounts = new int[nextSize];
		OOCFuture<BlockEntry>[] biggerFutures = new OOCFuture[nextSize];
		long[] biggerReleaseDueNanos = new long[nextSize];
		PackedUnpinHandle[] biggerReleaseHandles = new PackedUnpinHandle[nextSize];
		System.arraycopy(_allowances, 0, biggerAllowances, 0, _size);
		System.arraycopy(_counts, 0, biggerCounts, 0, _size);
		System.arraycopy(_futures, 0, biggerFutures, 0, _size);
		System.arraycopy(_releaseDueNanos, 0, biggerReleaseDueNanos, 0, _size);
		System.arraycopy(_releaseHandles, 0, biggerReleaseHandles, 0, _size);
		_allowances = biggerAllowances;
		_counts = biggerCounts;
		_futures = biggerFutures;
		_releaseDueNanos = biggerReleaseDueNanos;
		_releaseHandles = biggerReleaseHandles;
	}

	private void cancelRelease(int ix) {
		_releaseDueNanos[ix] = 0;
		PackedUnpinHandle handle = _releaseHandles[ix];
		if(handle != null) {
			_releaseHandles[ix] = null;
			handle.complete(false);
		}
	}

	private void releasePhysicalPin(OOCCacheImpl physical, MemoryAllowance allowance, PackedUnpinHandle handle) {
		OOCCache.UnpinHandle physicalHandle = physical.unpin(physicalEntry, allowance);
		if(physicalHandle.isCommitted()) {
			handle.complete(true);
			return;
		}
		physicalHandle.getCompletionFuture()
			.whenComplete((committed, ex) -> handle.complete(ex == null && Boolean.TRUE.equals(committed)));
	}

	private synchronized void removeFailedAllowance(MemoryAllowance allowance, OOCFuture<BlockEntry> future) {
		int ix = indexOf(allowance);
		if(ix >= 0 && _futures[ix] == future)
			removeAt(ix);
	}

	private void removeAt(int ix) {
		int last = --_size;
		_allowances[ix] = _allowances[last];
		_counts[ix] = _counts[last];
		_futures[ix] = _futures[last];
		_releaseDueNanos[ix] = _releaseDueNanos[last];
		_releaseHandles[ix] = _releaseHandles[last];
		_allowances[last] = null;
		_counts[last] = 0;
		_futures[last] = null;
		_releaseDueNanos[last] = 0;
		_releaseHandles[last] = null;
	}

	private record PackedRelease(MemoryAllowance allowance, PackedUnpinHandle handle) {
	}
}
