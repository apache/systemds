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
import static org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils.awaitUsedMemory;

import java.util.concurrent.TimeUnit;

import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.cache.io.OOCMatrixIOHandler;
import org.apache.sysds.runtime.ooc.cache.packed.OOCPackedCache;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils.RecordingOOCIOHandler;
import org.junit.Assert;
import org.junit.Test;

public class OOCPackedCacheTest {
	private static final long STREAM_ID = 41;
	private static final long BYTES = 1000;
	private static final long WAIT_TIMEOUT_SEC = 10;

	@Test
	public void testSmallTilesShareOnePhysicalPack() throws Exception {
		GlobalMemoryBroker broker = new GlobalMemoryBroker(1L << 32);
		SyncMemoryAllowance producer = new SyncMemoryAllowance(broker);
		producer.setTargetMemory(1L << 30);
		SyncMemoryAllowance reader = new SyncMemoryAllowance(broker);
		reader.setTargetMemory(1L << 30);
		OOCPackedCache cache = new OOCPackedCache(new OOCCacheImpl(new OOCMatrixIOHandler(), 1L << 30, 1L << 30),
			2 * BYTES, 10 * BYTES, -1, 0);
		try {
			BlockEntry[] entries = publishSmallTiles(cache, producer, STREAM_ID, 3);
			unpinAndFlush(cache, producer, entries);
			awaitUsedMemory(producer, 0, WAIT_TIMEOUT_SEC);

			Assert.assertEquals(1, cache.getPackGroupCount());
			OOCPackedCache.PackGroup group = cache.getPackGroup(STREAM_ID, 0);
			Assert.assertNotNull(group);
			Assert.assertEquals(3, group.size());

			BlockEntry first = cache.pin(STREAM_ID, 0, reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			BlockEntry second = cache.pin(STREAM_ID, 1, reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			Assert.assertEquals(1.0, scalar(first), 0.0);
			Assert.assertEquals(2.0, scalar(second), 0.0);
			Assert.assertEquals("Multiple logical pins in one pack should charge the physical pack once.", 3 * BYTES,
				reader.getUsedMemory());

			await(cache.unpin(first, reader), WAIT_TIMEOUT_SEC);
			Assert.assertEquals("The physical pack stays pinned while another logical pin remains.", 3 * BYTES,
				reader.getUsedMemory());
			await(cache.unpin(second, reader), WAIT_TIMEOUT_SEC);
			awaitUsedMemory(reader, 0, WAIT_TIMEOUT_SEC);
		}
		finally {
			cache.shutdown();
			producer.destroy();
			reader.destroy();
		}
	}

	@Test
	public void testLargeBlockBypassesPacking() throws Exception {
		GlobalMemoryBroker broker = new GlobalMemoryBroker(1L << 32);
		SyncMemoryAllowance producer = new SyncMemoryAllowance(broker);
		producer.setTargetMemory(1L << 30);
		SyncMemoryAllowance reader = new SyncMemoryAllowance(broker);
		reader.setTargetMemory(1L << 30);
		OOCPackedCache cache = new OOCPackedCache(new OOCCacheImpl(new OOCMatrixIOHandler(), 1L << 30, 1L << 30),
			2 * BYTES, 10 * BYTES, -1, 0);
		long largeBytes = 2 * BYTES;
		try {
			producer.reserveBlocking(largeBytes);
			BlockEntry entry = cache.putPinned(STREAM_ID, 0, value(5.0), largeBytes, producer);
			await(cache.unpin(entry, producer), WAIT_TIMEOUT_SEC);

			Assert.assertEquals(0, cache.getPackGroupCount());
			Assert.assertNull(cache.getPackGroup(STREAM_ID, 0));

			BlockEntry pinned = cache.pin(STREAM_ID, 0, reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			Assert.assertNotNull(pinned);
			Assert.assertEquals(5.0, scalar(pinned), 0.0);
			Assert.assertEquals(largeBytes, reader.getUsedMemory());
			await(cache.unpin(pinned, reader), WAIT_TIMEOUT_SEC);
			awaitUsedMemory(reader, 0, WAIT_TIMEOUT_SEC);
		}
		finally {
			cache.shutdown();
			producer.destroy();
			reader.destroy();
		}
	}

	@Test
	public void testPinPackExposesWholePack() throws Exception {
		GlobalMemoryBroker broker = new GlobalMemoryBroker(1L << 32);
		SyncMemoryAllowance producer = new SyncMemoryAllowance(broker);
		producer.setTargetMemory(1L << 30);
		SyncMemoryAllowance reader = new SyncMemoryAllowance(broker);
		reader.setTargetMemory(1L << 30);
		OOCPackedCache cache = new OOCPackedCache(new OOCCacheImpl(new OOCMatrixIOHandler(), 1L << 30, 1L << 30),
			2 * BYTES, 10 * BYTES, -1, 0);
		try {
			long[] tileIds = new long[] {2, 5};
			Object[] values = new Object[] {value(7.0), value(11.0)};
			long[] sizes = new long[] {BYTES, BYTES};
			producer.reserveBlocking(2 * BYTES);
			BlockEntry physical = cache.putSealedPackPinned(STREAM_ID, tileIds, values, sizes, 0, tileIds.length,
				producer);
			await(cache.unpin(physical, producer), WAIT_TIMEOUT_SEC);
			awaitUsedMemory(producer, 0, WAIT_TIMEOUT_SEC);

			OOCPackedCache.PackGroup group = cache.getPackGroup(STREAM_ID, 5);
			Assert.assertNotNull(group);
			Assert.assertEquals(2, group.size());
			Assert.assertEquals(2, group.index(0));
			Assert.assertEquals(5, group.index(1));

			OOCPackedCache.PackLease lease = cache.pinPack(group, reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);
			Assert.assertNotNull(lease);
			try {
				Assert.assertEquals(7.0, scalar((IndexedMatrixValue) lease.value(0)), 0.0);
				Assert.assertEquals(11.0, scalar((IndexedMatrixValue) lease.value(1)), 0.0);
				Assert.assertEquals(2 * BYTES, reader.getUsedMemory());
			}
			finally {
				lease.close();
			}
			awaitUsedMemory(reader, 0, WAIT_TIMEOUT_SEC);
		}
		finally {
			cache.shutdown();
			producer.destroy();
			reader.destroy();
		}
	}

	@Test
	public void testEvictedPackReplaysThroughLogicalPin() throws Exception {
		RecordingOOCIOHandler io = new RecordingOOCIOHandler();
		GlobalMemoryBroker broker = new GlobalMemoryBroker(1L << 32);
		SyncMemoryAllowance producer = new SyncMemoryAllowance(broker);
		producer.setTargetMemory(1L << 30);
		SyncMemoryAllowance reader = new SyncMemoryAllowance(broker);
		reader.setTargetMemory(1L << 30);
		OOCPackedCache cache = new OOCPackedCache(new OOCCacheImpl(io, 4 * BYTES, 0), 2 * BYTES, 10 * BYTES, -1, 0);
		try {
			BlockEntry[] entries = publishSmallTiles(cache, producer, STREAM_ID, 4);
			unpinAndFlush(cache, producer, entries);
			awaitUsedMemory(producer, 0, WAIT_TIMEOUT_SEC);
			await(() -> io.evictionCount() > 0 && cache.getOwnedCacheSize() == 0, WAIT_TIMEOUT_SEC);

			int readsBefore = io.readCount();
			BlockEntry pinned = cache.pin(STREAM_ID, 3, reader).get(WAIT_TIMEOUT_SEC, TimeUnit.SECONDS);

			Assert.assertNotNull(pinned);
			Assert.assertEquals(4.0, scalar(pinned), 0.0);
			Assert.assertTrue("Pinning an evicted logical tile should read the physical pack.",
				io.readCount() > readsBefore);
			Assert.assertEquals(4 * BYTES, reader.getUsedMemory());

			await(cache.unpin(pinned, reader), WAIT_TIMEOUT_SEC);
			awaitUsedMemory(reader, 0, WAIT_TIMEOUT_SEC);
		}
		finally {
			cache.shutdown();
			producer.destroy();
			reader.destroy();
		}
	}

	private static BlockEntry[] publishSmallTiles(OOCPackedCache cache, SyncMemoryAllowance producer, long streamId,
		int count) {
		BlockEntry[] entries = new BlockEntry[count];
		for(int i = 0; i < count; i++) {
			producer.reserveBlocking(BYTES);
			entries[i] = cache.putPinned(streamId, i, value(i + 1.0), BYTES, producer);
		}
		return entries;
	}

	private static void unpinAndFlush(OOCPackedCache cache, SyncMemoryAllowance producer, BlockEntry[] entries)
		throws Exception {
		OOCCache.UnpinHandle[] handles = new OOCCache.UnpinHandle[entries.length];
		for(int i = 0; i < entries.length; i++)
			handles[i] = cache.unpin(entries[i], producer);
		cache.flushPacks();
		for(OOCCache.UnpinHandle handle : handles)
			await(handle, WAIT_TIMEOUT_SEC);
	}

	private static IndexedMatrixValue value(double scalar) {
		return new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, scalar));
	}

	private static double scalar(BlockEntry entry) {
		return scalar((IndexedMatrixValue) entry.getData());
	}

	private static double scalar(IndexedMatrixValue value) {
		return value.getValue().get(0, 0);
	}
}
