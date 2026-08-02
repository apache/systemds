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

package org.apache.sysds.test.component.ooc.cache;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.OOCIOHandler;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.junit.Assert;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

public class OOCCacheTestUtils {

	public static void await(OOCCache.UnpinHandle handle, long timeout) throws Exception {
		if(!handle.isCommitted())
			handle.getCompletionFuture().get(timeout, TimeUnit.SECONDS);
	}

	public static void await(BooleanSupplier condition, long timeout) throws Exception {
		long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeout);
		while(!condition.getAsBoolean() && System.nanoTime() < deadline)
			Thread.sleep(1);
		Assert.assertTrue(condition.getAsBoolean());
	}

	public static void awaitUsedMemory(SyncMemoryAllowance allowance, long expected, long timeout) throws Exception {
		await(() -> allowance.getUsedMemory() == expected, timeout);
	}

	public static class RecordingOOCIOHandler implements OOCIOHandler {
		private final Map<BlockKey, Object> _spilled = new ConcurrentHashMap<>();
		private final AtomicInteger _evictions = new AtomicInteger();
		private final AtomicInteger _reads = new AtomicInteger();
		private volatile boolean _failReads;

		@Override
		public void shutdown() {
			_spilled.clear();
		}

		@Override
		public CompletableFuture<Void> scheduleEviction(BlockEntry block) {
			_spilled.put(block.getKey(), BlockEntryTestAccess.getDataUnsafe(block));
			_evictions.incrementAndGet();
			return CompletableFuture.completedFuture(null);
		}

		@Override
		public OOCFuture<BlockEntry> scheduleRead(BlockEntry block) {
			_reads.incrementAndGet();
			if(_failReads)
				return OOCFuture.failed(new RuntimeException("Injected read failure"));
			Object data = _spilled.get(block.getKey());
			if(data == null)
				return OOCFuture.completed(null);
			BlockEntryTestAccess.setDataUnsafe(block, data);
			return OOCFuture.completed(block);
		}

		@Override
		public void prioritizeRead(BlockKey key, double priority) {
		}

		@Override
		public CompletableFuture<Boolean> scheduleDeletion(BlockEntry block) {
			_spilled.remove(block.getKey());
			return CompletableFuture.completedFuture(true);
		}

		@Override
		public void registerSourceLocation(BlockKey key, OOCIOHandler.SourceBlockDescriptor descriptor) {
		}

		@Override
		public CompletableFuture<OOCIOHandler.SourceReadResult> scheduleSourceRead(
			OOCIOHandler.SourceReadRequest request) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		@Override
		public CompletableFuture<OOCIOHandler.SourceReadResult> continueSourceRead(
			OOCIOHandler.SourceReadContinuation continuation, long maxBytesInFlight) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		public int evictionCount() {
			return _evictions.get();
		}

		public int readCount() {
			return _reads.get();
		}

		public void failReads(boolean failReads) {
			_failReads = failReads;
		}

		public void reset() {
			_spilled.clear();
			_evictions.set(0);
			_reads.set(0);
			_failReads = false;
		}
	}
}
