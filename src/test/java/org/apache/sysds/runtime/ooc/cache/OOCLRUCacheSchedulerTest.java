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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class OOCLRUCacheSchedulerTest {
	private static final long ENTRY_SIZE = 1;
	private static final long WAIT_TIMEOUT_SEC = 5;

	private FakeIOHandler _handler;
	private OOCLRUCacheScheduler _scheduler;

	@Before
	public void setUp() {
		_handler = new FakeIOHandler();
		_scheduler = new OOCLRUCacheScheduler(_handler, 0, Long.MAX_VALUE);
	}

	@After
	public void tearDown() {
		if (_scheduler != null)
			_scheduler.shutdown();
		if (_handler != null)
			_handler.shutdown();
	}

	@Test
	public void testImmediateRequestPinsBlock() throws Exception {
		FakeIOHandler handler = new FakeIOHandler();
		OOCLRUCacheScheduler scheduler = new OOCLRUCacheScheduler(handler, Long.MAX_VALUE, Long.MAX_VALUE);
		try {
			BlockKey key = new BlockKey(1, 1);
			scheduler.put(key, new Object(), ENTRY_SIZE);
			Assert.assertEquals(ENTRY_SIZE, scheduler.getCacheSize());

			BlockEntry fetched = scheduler.request(key).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			Assert.assertTrue(fetched.isPinned());
			scheduler.unpin(fetched);
			Assert.assertEquals(ENTRY_SIZE, scheduler.getCacheSize());
		}
		finally {
			scheduler.shutdown();
			handler.shutdown();
		}
	}

	@Test
	public void testDeferredReadSingleBlock() throws Exception {
		BlockKey key = new BlockKey(1, 1);
		BlockEntry entry = putColdSourceBacked(key);
		Assert.assertEquals(0, _scheduler.getCacheSize());

		CompletableFuture<BlockEntry> future = _scheduler.request(key);
		Assert.assertFalse(future.isDone());
		Assert.assertEquals(1, _handler.getReadCount(key));
		Assert.assertEquals(ENTRY_SIZE, _scheduler.getCacheSize());

		_handler.completeRead(key);

		BlockEntry fetched = future.get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertTrue(fetched.isPinned());
		Assert.assertEquals(ENTRY_SIZE, _scheduler.getCacheSize());
		_scheduler.unpin(fetched);
		Assert.assertEquals(0, _scheduler.getCacheSize());
	}

	@Test
	public void testMergeOverlappingRequests() throws Exception {
		BlockKey key1 = new BlockKey(1, 1);
		BlockKey key2 = new BlockKey(1, 2);
		BlockKey key3 = new BlockKey(1, 3);
		putColdSourceBacked(key1);
		putColdSourceBacked(key2);
		putColdSourceBacked(key3);
		Assert.assertEquals(0, _scheduler.getCacheSize());

		CompletableFuture<List<BlockEntry>> reqA = _scheduler.request(List.of(key1, key2));
		CompletableFuture<List<BlockEntry>> reqB = _scheduler.request(List.of(key1, key3));

		Assert.assertEquals(1, _handler.getReadCount(key1));
		Assert.assertEquals(1, _handler.getReadCount(key2));
		Assert.assertEquals(1, _handler.getReadCount(key3));
		Assert.assertEquals(ENTRY_SIZE * 3, _scheduler.getCacheSize());
		Assert.assertFalse(reqA.isDone());
		Assert.assertFalse(reqB.isDone());

		_handler.completeRead(key1);
		_handler.completeRead(key2);

		List<BlockEntry> resA = reqA.get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertFalse(reqB.isDone());
		Assert.assertEquals(ENTRY_SIZE * 3, _scheduler.getCacheSize());

		_handler.completeRead(key3);
		List<BlockEntry> resB = reqB.get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertEquals(ENTRY_SIZE * 3, _scheduler.getCacheSize());

		resA.forEach(_scheduler::unpin);
		resB.forEach(_scheduler::unpin);
		Assert.assertEquals(0, _scheduler.getCacheSize());
	}

	@Test
	public void testPrioritizeReordersDeferredRequests() throws Exception {
		OOCLRUCacheScheduler scheduler = new OOCLRUCacheScheduler(_handler, 0, 0);
		try {
			BlockKey key1 = new BlockKey(1, 1);
			BlockKey key2 = new BlockKey(1, 2);
			BlockKey key3 = new BlockKey(1, 3);
			putColdSourceBacked(scheduler, key1);
			putColdSourceBacked(scheduler, key2);
			putColdSourceBacked(scheduler, key3);

			scheduler.request(List.of(key1));
			scheduler.request(List.of(key2));
			scheduler.request(List.of(key3));

			List<BlockKey> before = snapshotDeferredOrder(scheduler);
			Assert.assertEquals(List.of(key1, key2, key3), before);

			scheduler.prioritize(key3, 1);

			List<BlockKey> after = snapshotDeferredOrder(scheduler);
			Assert.assertEquals(List.of(key1, key3, key2), after);
		}
		finally {
			scheduler.shutdown();
		}
	}

	private BlockEntry putColdSourceBacked(BlockKey key) {
		return putColdSourceBacked(_scheduler, key);
	}

	private BlockEntry putColdSourceBacked(OOCLRUCacheScheduler scheduler, BlockKey key) {
		OOCIOHandler.SourceBlockDescriptor desc = new OOCIOHandler.SourceBlockDescriptor(
			"unused", Types.FileFormat.BINARY, new MatrixIndexes(1, 1), 0, 0, ENTRY_SIZE);
		BlockEntry entry = scheduler.putAndPinSourceBacked(key, new Object(), ENTRY_SIZE, desc);
		scheduler.unpin(entry);
		Assert.assertEquals(BlockState.COLD, entry.getState());
		return entry;
	}

	@SuppressWarnings("unchecked")
	private static List<BlockKey> snapshotDeferredOrder(OOCLRUCacheScheduler scheduler) throws Exception {
		Field field = OOCLRUCacheScheduler.class.getDeclaredField("_deferredReadRequests");
		field.setAccessible(true);
		Deque<Object> deque = (Deque<Object>) field.get(scheduler);
		List<BlockKey> order = new ArrayList<>();
		for (Object obj : deque) {
			Field entriesField = obj.getClass().getDeclaredField("_entries");
			entriesField.setAccessible(true);
			List<BlockEntry> entries = (List<BlockEntry>) entriesField.get(obj);
			order.add(entries.get(0).getKey());
		}
		return order;
	}

	private static class FakeIOHandler implements OOCIOHandler {
		private final Map<BlockKey, CompletableFuture<BlockEntry>> _readFutures = new HashMap<>();
		private final Map<BlockKey, BlockEntry> _readEntries = new HashMap<>();
		private final Map<BlockKey, AtomicInteger> _readCounts = new HashMap<>();

		@Override
		public void shutdown() {
			_readFutures.clear();
			_readEntries.clear();
			_readCounts.clear();
		}

		@Override
		public CompletableFuture<Void> scheduleEviction(BlockEntry block) {
			return CompletableFuture.completedFuture(null);
		}

		@Override
		public CompletableFuture<BlockEntry> scheduleRead(BlockEntry block) {
			CompletableFuture<BlockEntry> future = new CompletableFuture<>();
			_readFutures.put(block.getKey(), future);
			_readEntries.put(block.getKey(), block);
			_readCounts.computeIfAbsent(block.getKey(), k -> new AtomicInteger(0)).incrementAndGet();
			return future;
		}

		@Override
		public void prioritizeRead(BlockKey key, double priority) {}

		@Override
		public CompletableFuture<Boolean> scheduleDeletion(BlockEntry block) {
			return CompletableFuture.completedFuture(true);
		}

		@Override
		public void registerSourceLocation(BlockKey key, SourceBlockDescriptor descriptor) {
		}

		@Override
		public CompletableFuture<SourceReadResult> scheduleSourceRead(SourceReadRequest request) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		@Override
		public CompletableFuture<SourceReadResult> continueSourceRead(SourceReadContinuation continuation, long maxBytesInFlight) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		public int getReadCount(BlockKey key) {
			AtomicInteger ctr = _readCounts.get(key);
			return ctr == null ? 0 : ctr.get();
		}

		public void completeRead(BlockKey key) {
			CompletableFuture<BlockEntry> future = _readFutures.get(key);
			if (future == null)
				throw new IllegalStateException("No scheduled read for " + key);
			BlockEntry entry = _readEntries.get(key);
			if (entry == null)
				throw new IllegalStateException("No registered entry for " + key);
			entry.setDataUnsafe(new Object());
			future.complete(entry);
		}
	}
}
