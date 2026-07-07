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
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;

final class PackedUnpinHandle implements OOCCache.UnpinHandle {
	final BlockEntry entry;
	final MemoryAllowance allowance;
	final long bytes;
	final OOCFuture<Boolean> future;

	static PackedUnpinHandle committed(BlockEntry entry, MemoryAllowance allowance, long bytes) {
		return new PackedUnpinHandle(entry, allowance, bytes, true);
	}

	static PackedUnpinHandle pendingProducerTransfer(BlockEntry entry, MemoryAllowance allowance, long bytes) {
		return new PackedUnpinHandle(entry, allowance, bytes, false);
	}

	static PackedUnpinHandle delayedPhysicalRelease(BlockEntry entry, MemoryAllowance allowance) {
		return new PackedUnpinHandle(entry, allowance, entry.getSize(), false);
	}

	private PackedUnpinHandle(BlockEntry entry, MemoryAllowance allowance, long bytes, boolean committed) {
		this.entry = entry;
		this.allowance = allowance;
		this.bytes = bytes;
		future = committed ? OOCFuture.completed(true) : new OOCFuture<>();
	}

	@Override
	public BlockEntry entry() {
		return entry;
	}

	@Override
	public MemoryAllowance allowance() {
		return allowance;
	}

	@Override
	public long bytes() {
		return bytes;
	}

	@Override
	public boolean isCommitted() {
		return Boolean.TRUE.equals(future.getNow(false));
	}

	@Override
	public OOCFuture<Boolean> getCompletionFuture() {
		return future;
	}

	void complete(boolean committed) {
		future.complete(committed);
	}
}
