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

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.OOCIOHandler;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class OOCCacheImplTest {
	private static final long STREAM_ID = 7;
	private static final long BLOCK_ID = 3;
	private static final long BYTES = 1_000;
	private static final long WAIT_TIMEOUT_SEC = 10;

	private RecordingIOHandler _io;
	private GlobalMemoryBroker _broker;
	private SyncMemoryAllowance _producer;
	private SyncMemoryAllowance _reader;
	private OOCCacheImpl _cache;

	@Before
	public void setUp() {
		_io = new RecordingIOHandler();
		_broker = new GlobalMemoryBroker(8 * BYTES);
		_producer = new SyncMemoryAllowance(_broker, 4 * BYTES);
		_reader = new SyncMemoryAllowance(_broker, 4 * BYTES);
		_cache = new OOCCacheImpl(_io, 4 * BYTES, 4 * BYTES);
	}

	@After
	public void tearDown() {
		if(_cache != null)
			_cache.shutdown();
		if(_producer != null)
			_producer.destroy();
		if(_reader != null)
			_reader.destroy();
	}

	@Test
	public void testPinMissingEntryReturnsNullWithoutReservation() throws Exception {
		BlockEntry pinned = _cache.pin(new BlockKey(STREAM_ID, BLOCK_ID), _reader).get(WAIT_TIMEOUT_SEC,
			TimeUnit.SECONDS);

		Assert.assertNull(pinned);
		Assert.assertNull(_cache.pinIfLive(STREAM_ID, BLOCK_ID, _reader));
		Assert.assertEquals(0, _reader.getUsedMemory());
		Assert.assertEquals(0, _io.readCount());
	}

	@Test
	public void testResidentPinTransfersOwnershipBetweenCacheAndAllowance() throws Exception {
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);
		String payload = "resident";

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, payload, BYTES, _producer);
		Assert.assertEquals(0, _cache.getOwnedCacheSize());
		Assert.assertEquals(BYTES, _producer.getUsedMemory());

		await(_cache.unpin(entry, _producer));
		Assert.assertEquals(BYTES, _cache.getOwnedCacheSize());
		Assert.assertEquals(0, _producer.getUsedMemory());

		BlockEntry pinned = _cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertSame(entry, pinned);
		Assert.assertEquals(payload, pinned.getData());
		Assert.assertEquals(0, _cache.getOwnedCacheSize());
		Assert.assertEquals(BYTES, _reader.getUsedMemory());
		Assert.assertEquals(0, _io.readCount());

		await(_cache.unpin(pinned, _reader));
		Assert.assertEquals(BYTES, _cache.getOwnedCacheSize());
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	@Test
	public void testPinReloadsColdBackedEntry() throws Exception {
		useEvictingCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);
		String payload = "payload";

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, payload, BYTES, _producer);
		await(_cache.unpin(entry, _producer));
		waitFor(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null);
		Assert.assertEquals(0, _producer.getUsedMemory());

		BlockEntry pinned = _cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);

		Assert.assertNotNull(pinned);
		Assert.assertSame(entry, pinned);
		Assert.assertEquals(payload, pinned.getData());
		Assert.assertEquals(1, _io.readCount());
		Assert.assertEquals(BYTES, _reader.getUsedMemory());

		await(_cache.unpin(pinned, _reader));
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	@Test
	public void testPinIfLiveDoesNotReadColdBackedEntry() throws Exception {
		useEvictingCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);
		String payload = "cold";

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, payload, BYTES, _producer);
		await(_cache.unpin(entry, _producer));
		waitFor(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null);

		BlockEntry pinned = _cache.pinIfLive(STREAM_ID, BLOCK_ID, _reader);

		Assert.assertNull(pinned);
		Assert.assertEquals(0, _io.readCount());
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	@Test
	public void testDeferredUnpinCommitsWhenLimitsGrow() throws Exception {
		useZeroHardLimitCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, "deferred", BYTES, _producer);
		OOCCache.UnpinHandle deferred = _cache.unpin(entry, _producer);

		Assert.assertFalse(deferred.isCommitted());
		Assert.assertFalse(deferred.getCompletionFuture().isDone());
		Assert.assertEquals(BYTES, _producer.getUsedMemory());
		Assert.assertEquals(0, _cache.getOwnedCacheSize());

		_cache.updateLimits(BYTES, BYTES);
		deferred.getCompletionFuture().get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);

		Assert.assertTrue(deferred.isCommitted());
		Assert.assertEquals(0, _producer.getUsedMemory());
		Assert.assertEquals(BYTES, _cache.getOwnedCacheSize());
	}

	@Test
	public void testDeferredUnpinCanBeAdoptedBySameAllowance() throws Exception {
		useZeroHardLimitCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, "adopt", BYTES, _producer);
		OOCCache.UnpinHandle deferred = _cache.unpin(entry, _producer);

		BlockEntry repinned = _cache.pin(key, _producer).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);

		Assert.assertSame(entry, repinned);
		Assert.assertTrue(deferred.getCompletionFuture().isDone());
		Assert.assertFalse(deferred.isCommitted());
		Assert.assertEquals(BYTES, _producer.getUsedMemory());
		Assert.assertEquals(0, _cache.getOwnedCacheSize());

		OOCCache.UnpinHandle cleanup = _cache.unpin(repinned, _producer);
		_cache.updateLimits(BYTES, BYTES);
		cleanup.getCompletionFuture().get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertEquals(0, _producer.getUsedMemory());
	}

	@Test
	public void testDereferenceRemovesEntryAfterLastUnpin() throws Exception {
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, "drop", BYTES, _producer);

		Assert.assertEquals(0, _cache.dereference(entry));
		await(_cache.unpin(entry, _producer));

		Assert.assertEquals(0, _producer.getUsedMemory());
		Assert.assertNull(_cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS));
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	@Test
	public void testBackingReadFailureReleasesReservedBytes() throws Exception {
		useEvictingCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, "fail-read", BYTES, _producer);
		await(_cache.unpin(entry, _producer));
		waitFor(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null);

		_io.failReads(true);
		try {
			_cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			Assert.fail("A failed backing read must fail the pin future.");
		}
		catch(ExecutionException expected) {
			// expected
		}

		Assert.assertEquals(1, _io.readCount());
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	private void useEvictingCache() {
		_cache.shutdown();
		_io.reset();
		_cache = new OOCCacheImpl(_io, 4 * BYTES, 0);
	}

	private void useZeroHardLimitCache() {
		_cache.shutdown();
		_io.reset();
		_cache = new OOCCacheImpl(_io, 0, 0);
	}

	private static void await(OOCCache.UnpinHandle handle) throws Exception {
		if(!handle.isCommitted())
			handle.getCompletionFuture().get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
	}

	private static void waitFor(BooleanSupplier condition) throws Exception {
		long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(WAIT_TIMEOUT_SEC);
		while(!condition.getAsBoolean() && System.nanoTime() < deadline)
			Thread.sleep(1);
		Assert.assertTrue(condition.getAsBoolean());
	}

	private static final class RecordingIOHandler implements OOCIOHandler {
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
				return OOCFuture.failed(new IllegalStateException("read failed"));
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
		public void registerSourceLocation(BlockKey key, SourceBlockDescriptor descriptor) {
		}

		@Override
		public CompletableFuture<SourceReadResult> scheduleSourceRead(SourceReadRequest request) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		@Override
		public CompletableFuture<SourceReadResult> continueSourceRead(SourceReadContinuation continuation,
			long maxBytesInFlight) {
			return CompletableFuture.failedFuture(new UnsupportedOperationException());
		}

		private int evictionCount() {
			return _evictions.get();
		}

		private int readCount() {
			return _reads.get();
		}

		private void failReads(boolean failReads) {
			_failReads = failReads;
		}

		private void reset() {
			_spilled.clear();
			_evictions.set(0);
			_reads.set(0);
			_failReads = false;
		}
	}
}
