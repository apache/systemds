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

import static org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils.await;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils.RecordingOOCIOHandler;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class OOCCacheImplTest {
	private static final long STREAM_ID = 7;
	private static final long BLOCK_ID = 3;
	private static final long BYTES = 1_000;
	private static final long WAIT_TIMEOUT_SEC = 10;

	private RecordingOOCIOHandler _io;
	private GlobalMemoryBroker _broker;
	private SyncMemoryAllowance _producer;
	private SyncMemoryAllowance _reader;
	private OOCCacheImpl _cache;

	@Before
	public void setUp() {
		_io = new RecordingOOCIOHandler();
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

		await(_cache.unpin(entry, _producer), WAIT_TIMEOUT_SEC);
		Assert.assertEquals(BYTES, _cache.getOwnedCacheSize());
		Assert.assertEquals(0, _producer.getUsedMemory());

		BlockEntry pinned = _cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
		Assert.assertSame(entry, pinned);
		Assert.assertEquals(payload, pinned.getData());
		Assert.assertEquals(0, _cache.getOwnedCacheSize());
		Assert.assertEquals(BYTES, _reader.getUsedMemory());
		Assert.assertEquals(0, _io.readCount());

		await(_cache.unpin(pinned, _reader), WAIT_TIMEOUT_SEC);
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
		await(_cache.unpin(entry, _producer), WAIT_TIMEOUT_SEC);
		await(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null, WAIT_TIMEOUT_SEC);
		Assert.assertEquals(0, _producer.getUsedMemory());

		BlockEntry pinned = _cache.pin(key, _reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);

		Assert.assertNotNull(pinned);
		Assert.assertSame(entry, pinned);
		Assert.assertEquals(payload, pinned.getData());
		Assert.assertEquals(1, _io.readCount());
		Assert.assertEquals(BYTES, _reader.getUsedMemory());

		await(_cache.unpin(pinned, _reader), WAIT_TIMEOUT_SEC);
		Assert.assertEquals(0, _reader.getUsedMemory());
	}

	@Test
	public void testPinIfLiveDoesNotReadColdBackedEntry() throws Exception {
		useEvictingCache();
		BlockKey key = new BlockKey(STREAM_ID, BLOCK_ID);
		String payload = "cold";

		_producer.reserveBlocking(BYTES);
		BlockEntry entry = _cache.putPinned(key, payload, BYTES, _producer);
		await(_cache.unpin(entry, _producer), WAIT_TIMEOUT_SEC);
		await(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null, WAIT_TIMEOUT_SEC);

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
		await(_cache.unpin(entry, _producer), WAIT_TIMEOUT_SEC);

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
		await(_cache.unpin(entry, _producer), WAIT_TIMEOUT_SEC);
		await(() -> _io.evictionCount() == 1 && BlockEntryTestAccess.getDataUnsafe(entry) == null, WAIT_TIMEOUT_SEC);

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
}
