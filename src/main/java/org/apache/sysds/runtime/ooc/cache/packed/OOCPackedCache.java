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

import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.BlockState;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.collections.MaskedOnceArrayList;
import org.apache.sysds.runtime.ooc.cache.collections.SegmentedStreamTableList;
import org.apache.sysds.runtime.ooc.cache.io.OOCIOHandler;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.LongUnaryOperator;

public final class OOCPackedCache implements OOCCache {
	private static final long PACKED_STREAM_ID = CachingStream._streamSeq.getNextID();
	private static final long DEFAULT_PACK_THRESHOLD_BYTES = 1L << 18;
	private static final long DEFAULT_PACK_TARGET_BYTES = 1L << 19; // 512 KB tile packing
	private static final long DEFAULT_MAX_STAGING_BYTES = 1L << 26;
	private static final int DEFAULT_MAX_OPEN_BUILDERS = 64;
	private static final long DEFAULT_SEAL_DELAY_MS = 5;
	private static final long DEFAULT_PACK_RELEASE_DELAY_MS = 5;

	private final OOCCacheImpl _physical;
	private final long _packThresholdBytes;
	private final long _packTargetBytes;
	private final long _maxStagingBytes;
	private final int _maxOpenBuilders;
	private final long _sealDelayMs;
	private final long _packReleaseDelayMs;
	private final SegmentedStreamTableList<PackedCacheLocation> _locations;
	private final MaskedOnceArrayList<PackedPinState> _packedStates;
	private final ScheduledExecutorService _sealExecutor;
	private final ExecutorService _releaseExecutor;
	private final ConcurrentLinkedQueue<PackedPinState> _releaseQueue;
	private final AtomicBoolean _releaseRunning;
	private final AtomicBoolean _packedPolicyInstalled;
	private final AtomicInteger _nextPackedId;
	private final ArrayList<CopyOnWriteArrayList<LongUnaryOperator>> _logicalEvictionPolicies;

	private PackBuilder[] _builders;
	private long _stagingBytes;
	private int _openBuilderCount;
	private boolean _running;

	public OOCPackedCache(OOCIOHandler ioHandler, long hardLimit, long evictionLimit) {
		this(new OOCCacheImpl(ioHandler, hardLimit, evictionLimit), DEFAULT_PACK_THRESHOLD_BYTES,
			DEFAULT_PACK_TARGET_BYTES, DEFAULT_SEAL_DELAY_MS);
	}

	public OOCPackedCache(OOCCacheImpl physical) {
		this(physical, DEFAULT_PACK_THRESHOLD_BYTES, DEFAULT_PACK_TARGET_BYTES, DEFAULT_SEAL_DELAY_MS);
	}

	public OOCPackedCache(OOCCacheImpl physical, long packThresholdBytes, long packTargetBytes, long sealDelayMs) {
		this(physical, packThresholdBytes, packTargetBytes, sealDelayMs, DEFAULT_PACK_RELEASE_DELAY_MS);
	}

	public OOCPackedCache(OOCCacheImpl physical, long packThresholdBytes, long packTargetBytes, long sealDelayMs,
		long packReleaseDelayMs) {
		this(physical, packThresholdBytes, packTargetBytes, DEFAULT_MAX_STAGING_BYTES, DEFAULT_MAX_OPEN_BUILDERS,
			sealDelayMs, packReleaseDelayMs);
	}

	public OOCPackedCache(OOCCacheImpl physical, long packThresholdBytes, long packTargetBytes, long maxStagingBytes,
		int maxOpenBuilders, long sealDelayMs, long packReleaseDelayMs) {
		if(packThresholdBytes <= 0 || packTargetBytes < packThresholdBytes)
			throw new IllegalArgumentException(
				"Invalid pack sizes: threshold=" + packThresholdBytes + ", target=" + packTargetBytes);
		_physical = physical;
		_packThresholdBytes = packThresholdBytes;
		_packTargetBytes = packTargetBytes;
		_maxStagingBytes = Math.max(packTargetBytes, maxStagingBytes);
		_maxOpenBuilders = Math.max(1, maxOpenBuilders);
		_sealDelayMs = sealDelayMs;
		_packReleaseDelayMs = packReleaseDelayMs;
		_locations = new SegmentedStreamTableList<>();
		_packedStates = new MaskedOnceArrayList<>();
		_nextPackedId = new AtomicInteger();
		_releaseQueue = new ConcurrentLinkedQueue<>();
		_releaseRunning = new AtomicBoolean(false);
		_packedPolicyInstalled = new AtomicBoolean(false);
		_logicalEvictionPolicies = new ArrayList<>();
		_builders = new PackBuilder[16];
		_stagingBytes = 0;
		_openBuilderCount = 0;
		_running = true;
		_sealExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
			Thread t = new Thread(r, "ooc-pack-sealer");
			t.setDaemon(true);
			return t;
		});
		_releaseExecutor = Executors.newSingleThreadExecutor(r -> {
			Thread t = new Thread(r, "ooc-pack-release");
			t.setDaemon(true);
			return t;
		});
	}

	@Override
	public BlockEntry putPinned(long sId, long tId, Object data, long size, MemoryAllowance allowance) {
		if(size >= _packThresholdBytes)
			return _physical.putPinned(sId, tId, data, size, allowance);

		PackBuilder builder;
		int slot;
		synchronized(this) {
			checkRunning();
			builder = getOpenBuilder(sId, allowance, size);
			slot = appendToBuilder(builder, sId, tId, data, size);
		}

		BlockEntry logical = new BlockEntry(new BlockKey(sId, tId), size, data, BlockState.REMOVED);
		logical.pin();
		logical.setCacheMeta(new PendingLogicalPin(builder, slot));
		return logical;
	}

	public BlockEntry[] putPackPinned(long sId, long[] tIds, Object[] data, long[] sizes, int off, int len,
		MemoryAllowance allowance) {
		BlockEntry[] entries = new BlockEntry[len];
		synchronized(this) {
			checkRunning();
			for(int i = 0; i < len; i++) {
				int p = off + i;
				long tId = tIds[p];
				long size = sizes[p];
				if(size >= _packThresholdBytes) {
					entries[i] = _physical.putPinned(sId, tId, data[p], size, allowance);
					continue;
				}
				PackBuilder builder = getOpenBuilder(sId, allowance, size);
				int slot = appendToBuilder(builder, sId, tId, data[p], size);
				BlockEntry logical = new BlockEntry(new BlockKey(sId, tId), size, data[p], BlockState.REMOVED);
				logical.pin();
				logical.setCacheMeta(new PendingLogicalPin(builder, slot));
				entries[i] = logical;
			}
		}
		return entries;
	}

	public BlockEntry putSealedPackPinned(long sId, long[] tIds, Object[] data, long[] sizes, int off, int len,
		MemoryAllowance allowance) {
		long totalSize = 0;
		Object[] packedData = new Object[len];
		long[] packedSizes = new long[len];
		for(int i = 0; i < len; i++) {
			int p = off + i;
			packedData[i] = data[p];
			packedSizes[i] = sizes[p];
			totalSize += sizes[p];
		}

		synchronized(this) {
			checkRunning();
			BlockEntry physicalEntry = putSealedBlockPinned(new PackedBlock(packedData, packedSizes, totalSize),
				allowance);
			PackedPinState state = new PackedPinState(physicalEntry, sId,
				Arrays.stream(tIds).mapToInt(Math::toIntExact).toArray(), off, len, len);
			registerPackedState(state);
			for(int i = 0; i < len; i++)
				putLocation(new BlockKey(sId, tIds[off + i]), new SealedPackLocation(state, i));
			return physicalEntry;
		}
	}

	public PackGroup getPackGroup(long sId, long tId) {
		PackedCacheLocation location = getLocation(sId, tId);
		if(location instanceof PendingPackLocation pending)
			location = forceSeal(pending);
		return location instanceof SealedPackLocation packed ? packed.state().group : null;
	}

	public int getPackGroupCount() {
		return _nextPackedId.get();
	}

	public OOCFuture<PackLease> pinPack(PackGroup group, MemoryAllowance allowance) {
		return group.state.pin(_physical, allowance, false)
			.map(entry -> entry == null ? null : new PackLease(this, group, allowance));
	}

	@Override
	public OOCFuture<BlockEntry> pin(long sId, long tId, MemoryAllowance allowance) {
		PackedCacheLocation location = getLocation(sId, tId);
		if(location == null)
			return _physical.pin(sId, tId, allowance);
		if(location instanceof PendingPackLocation pending)
			location = forceSeal(pending);
		if(!(location instanceof SealedPackLocation packed))
			return _physical.pin(sId, tId, allowance);

		return packed.state().pin(_physical, allowance, false).map(physicalEntry -> {
			if(physicalEntry == null)
				return null;
			return createLogicalPin(new BlockKey(sId, tId), packed);
		});
	}

	@Override
	public OOCFuture<BlockEntry> pinAdmitted(long sId, long tId, MemoryAllowance allowance) {
		PackedCacheLocation location = getLocation(sId, tId);
		if(location == null)
			return _physical.pinAdmitted(sId, tId, allowance);
		if(location instanceof PendingPackLocation pending)
			location = forceSeal(pending);
		if(!(location instanceof SealedPackLocation packed))
			return _physical.pinAdmitted(sId, tId, allowance);

		return packed.state().pinAdmitted(_physical, allowance).map(physicalEntry -> {
			if(physicalEntry == null)
				return null;
			return createLogicalPin(new BlockKey(sId, tId), packed);
		});
	}

	@Override
	public BlockEntry pinIfLive(long sId, long tId, MemoryAllowance allowance) {
		PackedCacheLocation location = getLocation(sId, tId);
		if(location == null)
			return _physical.pinIfLive(sId, tId, allowance);
		if(location instanceof PendingPackLocation pending)
			location = forceSeal(pending);
		if(!(location instanceof SealedPackLocation packed))
			return _physical.pinIfLive(sId, tId, allowance);

		if(packed.state().pinIfLive(_physical, allowance) == null)
			return null;
		return createLogicalPin(new BlockKey(sId, tId), packed);
	}

	@Override
	public UnpinHandle unpin(BlockEntry entry, MemoryAllowance allowance) {
		Object meta = entry.getCacheMeta();
		if(meta instanceof PendingLogicalPin pending)
			return unpinPending(entry, pending, allowance);
		if(meta instanceof PackedLogicalPin packed)
			return unpinPacked(entry, packed, allowance);
		return _physical.unpin(entry, allowance);
	}

	@Override
	public int reference(BlockEntry entry) {
		Object meta = entry.getCacheMeta();
		if(meta instanceof PackedLogicalPin packed)
			return packed.location().retain();
		if(meta instanceof PendingLogicalPin pending)
			return referencePending(pending.builder(), pending.slot());
		return _physical.reference(entry);
	}

	@Override
	public int dereference(BlockEntry entry) {
		Object meta = entry.getCacheMeta();
		if(meta instanceof PackedLogicalPin packed)
			return releaseLocation(entry.getKey(), packed.location());
		if(meta instanceof PendingLogicalPin pending)
			return dereferencePending(pending.builder(), pending.slot());
		return _physical.dereference(entry);
	}

	@Override
	public int dereference(BlockKey key) {
		PackedCacheLocation location = getLocation(key.getStreamId(), key.getSequenceNumber());
		if(location == null)
			return _physical.dereference(key);
		if(location instanceof PendingPackLocation pending)
			return dereferencePending(pending.builder(), pending.slot());
		if(!(location instanceof SealedPackLocation packed))
			return _physical.dereference(key);
		return releaseLocation(key, packed);
	}

	/**
	 * References/dereferences on tiles in open builders are counted on the builder slot instead of forcing a seal, so
	 * pipelined consumers that park references (state tables, store readers) do not fragment packs into per-tile
	 * physical entries. Slot counts carry over into the SealedPackLocation at seal time. Only physical access (pin,
	 * pack group) forces a seal.
	 */
	private synchronized int referencePending(PackBuilder builder, int slot) {
		if(!builder.sealed)
			return builder.retainSlot(slot);
		PackedCacheLocation location = getLocation(builder.streamIds[slot], builder.tileIds[slot]);
		if(!(location instanceof SealedPackLocation packed))
			throw new IllegalStateException("Cannot retain a forgotten packed location.");
		return packed.retain();
	}

	private synchronized int dereferencePending(PackBuilder builder, int slot) {
		BlockKey key = new BlockKey(builder.streamIds[slot], builder.tileIds[slot]);
		if(builder.sealed) {
			PackedCacheLocation location = getLocation(key.getStreamId(), key.getSequenceNumber());
			return location instanceof SealedPackLocation packed ? releaseLocation(key, packed) : 0;
		}
		int references = builder.releaseSlot(slot);
		if(references == 0)
			clearLocation(key);
		return references;
	}

	@Override
	public void updateLimits(long hardLimit, long evictionLimit) {
		_physical.updateLimits(hardLimit, evictionLimit);
	}

	@Override
	public void addEvictionPolicy(long streamId, LongUnaryOperator scoreFn) {
		_physical.addEvictionPolicy(streamId, scoreFn);
		addLogicalEvictionPolicy(streamId, scoreFn);
		if(_packedPolicyInstalled.compareAndSet(false, true))
			_physical.addEvictionPolicy(PACKED_STREAM_ID, this::scorePackedBlock);
	}

	@Override
	public long getOwnedCacheSize() {
		return _physical.getOwnedCacheSize();
	}

	@Override
	public synchronized void shutdown() {
		if(!_running)
			return;
		_running = false;
		for(PackBuilder builder : _builders)
			if(builder != null)
				sealBuilder(builder);
		_physical.updateLimits(Long.MAX_VALUE, Long.MAX_VALUE);
		_sealExecutor.shutdownNow();
		_releaseExecutor.shutdown();
		awaitReleaseExecutor();
		drainReleaseQueue();
		_physical.shutdown();
	}

	private void awaitReleaseExecutor() {
		try {
			_releaseExecutor.awaitTermination(Math.max(100, _packReleaseDelayMs * 2), TimeUnit.MILLISECONDS);
		}
		catch(InterruptedException ex) {
			Thread.currentThread().interrupt();
		}
	}

	private void drainReleaseQueue() {
		PackedPinState state;
		while((state = _releaseQueue.poll()) != null) {
			state.clearReleaseQueued();
			state.releaseDuePins(_physical, Long.MAX_VALUE);
		}
	}

	public synchronized void flushPacks() {
		for(PackBuilder builder : _builders)
			if(builder != null)
				sealBuilder(builder);
	}

	private UnpinHandle unpinPending(BlockEntry entry, PendingLogicalPin pin, MemoryAllowance allowance) {
		if(entry.fastUnpin()) {
			allowance.release(entry.getSize());
			return PackedUnpinHandle.committed(entry, allowance, entry.getSize());
		}
		synchronized(this) {
			if(entry.getPinCount() > 1) {
				entry.unpin();
				allowance.release(entry.getSize());
				return PackedUnpinHandle.committed(entry, allowance, entry.getSize());
			}
			entry.unpin();
			entry.setCacheMeta(null);
			PackedUnpinHandle handle = pin.builder().unpinProducer(entry, pin.slot(), allowance);
			if(pin.builder().sealed && pin.builder().activePins == 0)
				pin.builder().transferProducerOwnership(_physical);
			scheduleSeal(pin.builder());
			return handle;
		}
	}

	private UnpinHandle unpinPacked(BlockEntry entry, PackedLogicalPin pin, MemoryAllowance allowance) {
		if(entry.fastUnpin())
			return PackedUnpinHandle.committed(entry, allowance, entry.getSize());
		if(entry.getPinCount() > 1) {
			entry.unpin();
			return PackedUnpinHandle.committed(entry, allowance, entry.getSize());
		}
		entry.unpin();
		entry.setCacheMeta(null);
		return pin.location().state().unpin(this, _packReleaseDelayMs, allowance);
	}

	void enqueueRelease(PackedPinState state) {
		if(!_running)
			return;
		if(state.markReleaseQueued()) {
			_releaseQueue.offer(state);
			scheduleReleaseMaintenance();
		}
	}

	private void enqueueReleaseNoSchedule(PackedPinState state) {
		if(state.markReleaseQueued())
			_releaseQueue.offer(state);
	}

	private void scheduleReleaseMaintenance() {
		if(!_releaseRunning.compareAndSet(false, true))
			return;
		_releaseExecutor.execute(this::runReleaseMaintenance);
	}

	private void runReleaseMaintenance() {
		try {
			while(_running) {
				long nextDueNanos = Long.MAX_VALUE;
				ArrayList<PackedPinState> delayed = null;
				PackedPinState state;
				long nowNanos = System.nanoTime();
				while((state = _releaseQueue.poll()) != null) {
					state.clearReleaseQueued();
					long stateNextDue = state.releaseDuePins(_physical, nowNanos);
					if(stateNextDue != Long.MAX_VALUE) {
						if(delayed == null)
							delayed = new ArrayList<>();
						delayed.add(state);
						nextDueNanos = Math.min(nextDueNanos, stateNextDue);
					}
				}
				if(delayed != null)
					for(PackedPinState delayedState : delayed)
						enqueueReleaseNoSchedule(delayedState);
				if(nextDueNanos == Long.MAX_VALUE)
					return;
				long waitNanos = nextDueNanos - System.nanoTime();
				if(waitNanos > 0)
					LockSupport.parkNanos(waitNanos);
			}
		}
		finally {
			_releaseRunning.set(false);
			if(_running && !_releaseQueue.isEmpty())
				scheduleReleaseMaintenance();
		}
	}

	private SealedPackLocation forceSeal(PendingPackLocation pending) {
		synchronized(this) {
			sealBuilder(pending.builder());
			PackedCacheLocation location = getLocation(pending.builder().streamIds[pending.slot()],
				pending.builder().tileIds[pending.slot()]);
			return (SealedPackLocation) location;
		}
	}

	private static BlockEntry createLogicalPin(BlockKey logicalKey, SealedPackLocation location) {
		PackedBlock block = (PackedBlock) location.state().physicalEntry.getDataUnsafe();
		Object data = block.values[location.slot()];
		long size = block.sizes[location.slot()];
		BlockEntry logical = new BlockEntry(logicalKey, size, data, BlockState.REMOVED);
		logical.pin();
		logical.setCacheMeta(new PackedLogicalPin(location));
		return logical;
	}

	private int releaseLocation(BlockKey key, SealedPackLocation location) {
		int references = location.release();
		if(references > 0)
			return references;
		if(!clearLocation(key))
			return 0;
		if(location.state().forgetLocation()) {
			_packedStates.clear((int) location.state().physicalEntry.getKey().getSequenceNumber());
			return _physical.dereference(location.state().physicalEntry);
		}
		return 0;
	}

	private PackBuilder getOpenBuilder(long streamId, MemoryAllowance allowance, long nextSize) {
		int sid = (int) streamId;
		PackBuilder builder = sid < _builders.length ? _builders[sid] : null;
		if(builder != null && (builder.sealed || builder.allowance != allowance)) {
			sealBuilder(builder);
			builder = null;
		}
		if(builder != null)
			return builder;
		while(!canOpenBuilder(nextSize)) {
			PackBuilder largest = findLargestOpenBuilder();
			if(largest == null)
				break;
			sealBuilder(largest);
		}
		ensureBuilderCapacity(sid);
		builder = new PackBuilder(sid, allowance, _packTargetBytes);
		_builders[sid] = builder;
		_openBuilderCount++;
		return builder;
	}

	private boolean canOpenBuilder(long nextSize) {
		return _openBuilderCount < _maxOpenBuilders && _stagingBytes + nextSize <= _maxStagingBytes;
	}

	private int appendToBuilder(PackBuilder builder, long streamId, long tileId, Object data, long size) {
		int slot = builder.append(streamId, tileId, data, size);
		_stagingBytes += size;
		putLocation(new BlockKey(streamId, tileId), new PendingPackLocation(builder, slot));
		if(builder.getBytes() >= builder.packTargetBytes)
			sealBuilder(builder);
		else
			enforceStagingBudget();
		return slot;
	}

	private void enforceStagingBudget() {
		while(_stagingBytes > _maxStagingBytes || _openBuilderCount > _maxOpenBuilders) {
			PackBuilder builder = findLargestOpenBuilder();
			if(builder == null)
				return;
			sealBuilder(builder);
		}
	}

	private PackBuilder findLargestOpenBuilder() {
		PackBuilder largest = null;
		for(PackBuilder builder : _builders)
			if(builder != null && !builder.sealed && (largest == null || builder.getBytes() > largest.getBytes()))
				largest = builder;
		return largest;
	}

	private void sealBuilder(PackBuilder builder) {
		if(builder.sealed || builder.count == 0)
			return;
		builder.sealed = true;
		_stagingBytes -= builder.getBytes();
		_openBuilderCount--;
		if(builder.streamSlot >= 0 && builder.streamSlot < _builders.length && _builders[builder.streamSlot] == builder)
			_builders[builder.streamSlot] = null;

		PackedBlock block = builder.createBlock();
		BlockEntry physicalEntry = putSealedBlockPinned(block, builder.allowance);
		int liveSlots = builder.countLiveSlots();
		PackedPinState state = new PackedPinState(physicalEntry, builder.streamIds[0],
			Arrays.stream(builder.tileIds).mapToInt(Math::toIntExact).toArray(), 0, builder.count, liveSlots);
		builder.state = state;
		if(liveSlots > 0)
			registerPackedState(state);

		// slots forgotten while pending stay in the physical pack but get no location
		for(int i = 0; i < builder.count; i++)
			if(builder.refCounts[i] > 0)
				putLocation(new BlockKey(builder.streamIds[i], builder.tileIds[i]),
					new SealedPackLocation(state, i, builder.refCounts[i]));

		if(builder.activePins == 0)
			builder.transferProducerOwnership(_physical);
		if(liveSlots == 0)
			_physical.dereference(physicalEntry);
	}

	private BlockEntry putSealedBlockPinned(PackedBlock block, MemoryAllowance allowance) {
		BlockKey packedKey = new BlockKey(PACKED_STREAM_ID, _nextPackedId.getAndIncrement());
		return _physical.putPinned(packedKey, block, block.totalSize, allowance);
	}

	private void registerPackedState(PackedPinState state) {
		_packedStates.put((int) state.physicalEntry.getKey().getSequenceNumber(), state);
	}

	private long scorePackedBlock(long packId) {
		PackedPinState state = _packedStates.get((int) packId);
		if(state == null)
			return packId;
		PackGroup group = state.group;
		CopyOnWriteArrayList<LongUnaryOperator> policies = group.streamId < _logicalEvictionPolicies
			.size() ? _logicalEvictionPolicies.get((int) group.streamId) : null;
		if(policies == null || policies.isEmpty())
			return packId;
		long score = Long.MAX_VALUE;
		for(int i = 0; i < group.size(); i++) {
			long tileId = group.index(i);
			for(LongUnaryOperator policy : policies)
				score = Math.min(score, policy.applyAsLong(tileId));
		}
		return score;
	}

	private synchronized void addLogicalEvictionPolicy(long streamId, LongUnaryOperator scoreFn) {
		int sid = (int) streamId;
		while(sid >= _logicalEvictionPolicies.size())
			_logicalEvictionPolicies.add(null);
		CopyOnWriteArrayList<LongUnaryOperator> policies = _logicalEvictionPolicies.get(sid);
		if(policies == null) {
			policies = new CopyOnWriteArrayList<>();
			_logicalEvictionPolicies.set(sid, policies);
		}
		policies.add(scoreFn);
	}

	private void scheduleSeal(PackBuilder builder) {
		if(builder.sealScheduled || builder.sealed || _sealDelayMs < 0)
			return;
		builder.sealScheduled = true;
		_sealExecutor.schedule(() -> {
			synchronized(OOCPackedCache.this) {
				builder.sealScheduled = false;
				sealBuilder(builder);
			}
		}, _sealDelayMs, TimeUnit.MILLISECONDS);
	}

	private PackedCacheLocation getLocation(long sId, long tId) {
		MaskedOnceArrayList<PackedCacheLocation> stream = _locations.get(sId);
		return stream == null ? null : stream.get((int) tId);
	}

	private void putLocation(BlockKey key, PackedCacheLocation location) {
		_locations.getOrCreate(key.getStreamId()).put((int) key.getSequenceNumber(), location);
	}

	private boolean clearLocation(BlockKey key) {
		MaskedOnceArrayList<PackedCacheLocation> stream = _locations.get(key.getStreamId());
		return stream != null && stream.clear((int) key.getSequenceNumber());
	}

	private void ensureBuilderCapacity(int streamId) {
		if(streamId < _builders.length)
			return;
		int len = _builders.length;
		while(streamId >= len)
			len <<= 1;
		PackBuilder[] bigger = new PackBuilder[len];
		System.arraycopy(_builders, 0, bigger, 0, _builders.length);
		_builders = bigger;
	}

	private void checkRunning() {
		if(!_running)
			throw new IllegalStateException("Cache has been shut down.");
	}

	public static final class PackGroup {
		private final PackedPinState state;
		private final long streamId;
		private final int firstIndex;
		private final int[] indices;
		private final int size;

		PackGroup(PackedPinState state, long streamId, int[] tileIds, int off, int size) {
			this.state = state;
			this.streamId = streamId;
			this.size = size;
			firstIndex = tileIds[off];
			boolean contiguous = true;
			for(int i = 1; i < size; i++) {
				if(tileIds[off + i] != firstIndex + i) {
					contiguous = false;
					break;
				}
			}
			if(contiguous)
				indices = null;
			else {
				indices = new int[size];
				System.arraycopy(tileIds, off, indices, 0, size);
			}
		}

		public int id() {
			return (int) state.physicalEntry.getKey().getSequenceNumber();
		}

		public long streamId() {
			return streamId;
		}

		public int size() {
			return size;
		}

		public int index(int slot) {
			if(slot < 0 || slot >= size)
				throw new IndexOutOfBoundsException("Invalid pack slot: " + slot);
			return indices == null ? firstIndex + slot : indices[slot];
		}
	}

	public static final class PackLease implements AutoCloseable {
		private final OOCPackedCache owner;
		private final PackGroup group;
		private final MemoryAllowance allowance;
		private boolean open;

		private PackLease(OOCPackedCache owner, PackGroup group, MemoryAllowance allowance) {
			this.owner = owner;
			this.group = group;
			this.allowance = allowance;
			open = true;
		}

		public PackGroup group() {
			return group;
		}

		public int size() {
			return group.size();
		}

		public int index(int slot) {
			return group.index(slot);
		}

		public Object value(int slot) {
			if(!open)
				throw new IllegalStateException("Pack lease is closed");
			PackedBlock block = (PackedBlock) group.state.physicalEntry.getData();
			return block.values[slot];
		}

		@Override
		public void close() {
			if(!open)
				return;
			open = false;
			group.state.unpin(owner, owner._packReleaseDelayMs, allowance);
		}
	}
}
