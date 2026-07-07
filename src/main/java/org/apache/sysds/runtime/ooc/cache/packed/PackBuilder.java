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

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class PackBuilder {
	final int streamSlot;
	final MemoryAllowance allowance;
	final long packTargetBytes;
	final List<PackedUnpinHandle> deferredUnpins = new ArrayList<>();
	long[] streamIds = new long[16];
	long[] tileIds = new long[16];
	private Object[] values = new Object[16];
	long[] sizes = new long[16];
	int[] refCounts = new int[16];
	long bytes;
	int count;
	int activePins;
	boolean sealed;
	boolean sealScheduled;
	PackedPinState state;
	private boolean _producerTransferred;

	PackBuilder(int streamSlot, MemoryAllowance allowance, long packTargetBytes) {
		this.streamSlot = streamSlot;
		this.allowance = allowance;
		this.packTargetBytes = packTargetBytes;
	}

	int append(long streamId, long tileId, Object value, long size) {
		ensureCapacity(count + 1);
		int slot = count++;
		streamIds[slot] = streamId;
		tileIds[slot] = tileId;
		values[slot] = value;
		sizes[slot] = size;
		refCounts[slot] = 1;
		bytes += size;
		activePins++;
		return slot;
	}

	int retainSlot(int slot) {
		int references = refCounts[slot];
		if(references <= 0)
			throw new IllegalStateException("Cannot retain a forgotten packed location.");
		return refCounts[slot] = references + 1;
	}

	int releaseSlot(int slot) {
		int references = refCounts[slot];
		if(references <= 0)
			return 0;
		return refCounts[slot] = references - 1;
	}

	int countLiveSlots() {
		int live = 0;
		for(int i = 0; i < count; i++)
			if(refCounts[i] > 0)
				live++;
		return live;
	}

	long getBytes() {
		return bytes;
	}

	PackedBlock createBlock() {
		return new PackedBlock(Arrays.copyOf(values, count), Arrays.copyOf(sizes, count), bytes);
	}

	PackedUnpinHandle unpinProducer(BlockEntry entry, int slot, MemoryAllowance owner) {
		activePins--;
		PackedUnpinHandle handle = PackedUnpinHandle.pendingProducerTransfer(entry, owner, sizes[slot]);
		deferredUnpins.add(handle);
		return handle;
	}

	void transferProducerOwnership(OOCCacheImpl physical) {
		if(state == null || physical == null || _producerTransferred)
			return;
		_producerTransferred = true;
		OOCCache.UnpinHandle physicalUnpin = physical.unpin(state.physicalEntry, allowance);
		if(physicalUnpin.isCommitted()) {
			completeDeferredUnpins(true);
			return;
		}
		physicalUnpin.getCompletionFuture()
			.whenComplete((committed, ex) -> completeDeferredUnpins(ex == null && committed));
	}

	private void ensureCapacity(int minSize) {
		if(minSize <= values.length)
			return;
		int len = values.length;
		while(minSize > len)
			len <<= 1;
		streamIds = Arrays.copyOf(streamIds, len);
		tileIds = Arrays.copyOf(tileIds, len);
		values = Arrays.copyOf(values, len);
		sizes = Arrays.copyOf(sizes, len);
		refCounts = Arrays.copyOf(refCounts, len);
	}

	private void completeDeferredUnpins(boolean committed) {
		for(PackedUnpinHandle handle : deferredUnpins)
			handle.complete(committed);
		deferredUnpins.clear();
	}
}
