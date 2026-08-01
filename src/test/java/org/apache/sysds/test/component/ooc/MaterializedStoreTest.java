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

package org.apache.sysds.test.component.ooc;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.runtime.ooc.store.IndexedMaterializedStoreReader;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.store.CountingLiveness;
import org.apache.sysds.runtime.ooc.store.OOCStreamMaterializer;
import org.apache.sysds.runtime.ooc.store.OrderedMaterializedStoreReader;
import org.apache.sysds.runtime.ooc.store.SequentialAccessPattern;
import org.apache.sysds.runtime.ooc.store.StoreBackedStream;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class MaterializedStoreTest {
	private static final long MEMORY_LIMIT = 100_000_000;
	private static final long WAIT_SECONDS = 10;
	private static final long TILE_BYTES = new MatrixBlock(4, 4, 1.0).getExactSerializedSize();

	private GlobalMemoryBroker _broker;
	private SyncMemoryAllowance _producer;
	private SyncMemoryAllowance _materializerAllowance;
	private SyncMemoryAllowance _readerAllowance;
	private OOCCacheImpl _cache;
	private MaterializedStore<IndexedMatrixValue> _store;

	@Before
	public void setUp() {
		_broker = new GlobalMemoryBroker(1_000_000_000);
		_producer = new SyncMemoryAllowance(_broker);
		_materializerAllowance = new SyncMemoryAllowance(_broker);
		_readerAllowance = new SyncMemoryAllowance(_broker);
		_producer.setTargetMemory(MEMORY_LIMIT);
		_materializerAllowance.setTargetMemory(MEMORY_LIMIT);
		_readerAllowance.setTargetMemory(MEMORY_LIMIT);
		_cache = new OOCCacheImpl(new OOCCacheTestUtils.RecordingOOCIOHandler(), MEMORY_LIMIT, MEMORY_LIMIT);
		_store = new MaterializedStore<>(_cache, CachingStream._streamSeq.getNextID());
	}

	@After
	public void tearDown() {
		_store.close();
		_cache.shutdown();
		_producer.destroy();
		_materializerAllowance.destroy();
		_readerAllowance.destroy();
	}

	@Test
	public void testMaterializationReadersAndForgetting() throws Exception {
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance);
		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(0, 1.0), null, _producer, TILE_BYTES));
		materializer.accept(new OOCStream.SimpleQueueCallback<>(tile(1, 2.0), null));
		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(2, 3.0), null, _producer, TILE_BYTES));
		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);

		Assert.assertEquals(0, _producer.getUsedMemory());
		Assert.assertEquals(0, _materializerAllowance.getUsedMemory());
		Assert.assertEquals(3, _store.size());

		StoreBackedStream<IndexedMatrixValue> ordered = new StoreBackedStream<>(
			_store.openReader(new SequentialAccessPattern(3), _readerAllowance, 2, false));
		IndexedMaterializedStoreReader<IndexedMatrixValue> indexed = _store
			.openIndexedReader(new CountingLiveness(3, 1));
		_store.sealReaders();

		int index = 0;
		IndexedMatrixValue value;
		while((value = ordered.dequeue()) != null)
			Assert.assertEquals(++index, value.getIndexes().getRowIndex());
		Assert.assertEquals(3, index);
		Assert.assertTrue(_cache.getOwnedCacheSize() > 0);

		for(index = 2; index > 0; index--) {
			try(StoreLease<IndexedMatrixValue> lease = indexed.request(index, _readerAllowance).get(WAIT_SECONDS,
				TimeUnit.SECONDS)) {
				Assert.assertEquals(index + 1L, lease.value().getIndexes().getRowIndex());
			}
		}
		Assert.assertTrue(_cache.getOwnedCacheSize() > 0);
		indexed.close();
		OOCCacheTestUtils.await(() -> _cache.getOwnedCacheSize() == 0, WAIT_SECONDS);
		Assert.assertEquals(0, _readerAllowance.getUsedMemory());
	}

	@Test
	public void testOrderedReaderRetries() throws Exception {
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance);
		for(int i = 0; i < 2; i++) {
			_producer.reserveBlocking(TILE_BYTES);
			materializer.accept(new InMemoryQueueCallback(tile(i, i + 1.0), null, _producer, TILE_BYTES));
		}
		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);

		_readerAllowance.destroy();
		_readerAllowance = new SyncMemoryAllowance(_broker, TILE_BYTES);
		_readerAllowance.setTargetMemory(TILE_BYTES);
		OrderedMaterializedStoreReader<IndexedMatrixValue> reader = _store.openReader(new SequentialAccessPattern(2),
			_readerAllowance, 2, false);
		_store.sealReaders();

		Assert.assertTrue(reader.hasNext());
		try(StoreLease<IndexedMatrixValue> first = reader.next()) {
			Assert.assertEquals(1L, first.value().getIndexes().getRowIndex());
		}
		try(StoreLease<IndexedMatrixValue> second = reader.next()) {
			Assert.assertEquals(2L, second.value().getIndexes().getRowIndex());
		}
		reader.close();
		Assert.assertEquals(0, _readerAllowance.getUsedMemory());
	}

	@Test
	public void testSoftOrderingReturnsReadyRequestFirst() throws Exception {
		MatrixBlock largeBlock = new MatrixBlock(16, 16, 1.0);
		long largeBytes = largeBlock.getExactSerializedSize();
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance);
		_producer.reserveBlocking(largeBytes);
		materializer.accept(new InMemoryQueueCallback(new IndexedMatrixValue(new MatrixIndexes(1, 1), largeBlock), null,
			_producer, largeBytes));
		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(1, 2.0), null, _producer, TILE_BYTES));
		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);

		_readerAllowance.destroy();
		_readerAllowance = new SyncMemoryAllowance(_broker, largeBytes);
		_readerAllowance.setTargetMemory(largeBytes);
		long heldBytes = largeBytes - TILE_BYTES;
		_readerAllowance.reserveBlocking(heldBytes);
		OrderedMaterializedStoreReader<IndexedMatrixValue> reader = _store.openReader(new SequentialAccessPattern(2),
			_readerAllowance, 2);
		_store.sealReaders();

		Assert.assertTrue(reader.hasNext());
		try(StoreLease<IndexedMatrixValue> first = reader.next()) {
			Assert.assertEquals(2L, first.value().getIndexes().getRowIndex());
		}
		_readerAllowance.release(heldBytes);
		try(StoreLease<IndexedMatrixValue> second = reader.next()) {
			Assert.assertEquals(1L, second.value().getIndexes().getRowIndex());
		}
		Assert.assertFalse(reader.hasNext());
		reader.close();
		Assert.assertEquals(0, _readerAllowance.getUsedMemory());
	}

	@Test
	public void testDirectRequests() throws Exception {
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance);
		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(0, 1.0), null, _producer, TILE_BYTES));
		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);

		IndexedMaterializedStoreReader<IndexedMatrixValue> reader = _store
			.openIndexedReader(new CountingLiveness(1, 1));
		_store.sealReaders();

		try(StoreLease<IndexedMatrixValue> published = _store.requestPublished(0, _readerAllowance).get(WAIT_SECONDS,
			TimeUnit.SECONDS)) {
			Assert.assertEquals(1L, published.value().getIndexes().getRowIndex());
		}
		StoreLease<IndexedMatrixValue> live = reader.requestIfLive(0, _readerAllowance);
		Assert.assertNotNull(live);
		StoreLease<IndexedMatrixValue> retained = live.retain();
		live.close();
		Assert.assertEquals(TILE_BYTES, _readerAllowance.getUsedMemory());
		Assert.assertEquals(1L, retained.value().getIndexes().getRowIndex());
		retained.close();
		Assert.assertEquals(0, _readerAllowance.getUsedMemory());
		reader.close();
		Assert.assertTrue(reader.isClosed());
	}

	@Test
	public void testCompletionMissingPublications() throws Exception {
		AtomicInteger failures = new AtomicInteger();
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance, List.of(callback -> {
				if(callback.isFailure())
					failures.incrementAndGet();
			}));
		for(int index : new int[] {0, 2}) {
			_producer.reserveBlocking(TILE_BYTES);
			materializer.accept(new InMemoryQueueCallback(tile(index, 1.0), null, _producer, TILE_BYTES));
		}
		materializer.accept(OOCStream.eos(null));

		try {
			materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);
			Assert.fail("Completion must reject a missing publication index");
		}
		catch(ExecutionException ex) {
			Assert.assertTrue(ex.getCause() instanceof IllegalStateException);
		}
		Assert.assertEquals(1, failures.get());
		Assert.assertEquals(0, _producer.getUsedMemory());
	}

	@Test
	public void testLiveCallbackKeepsPublicationPinned() throws Exception {
		AtomicReference<OOCStream.QueueCallback<IndexedMatrixValue>> retained = new AtomicReference<>();
		AtomicInteger eos = new AtomicInteger();
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance, List.of(callback -> {
				if(callback.isEos())
					eos.incrementAndGet();
				else
					retained.set(callback.keepOpen());
			}));

		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(0, 1.0), null, _producer, TILE_BYTES));
		Assert.assertEquals(TILE_BYTES, _producer.getUsedMemory());
		Assert.assertNotNull(retained.get());
		retained.get().close();
		retained.get().close();
		Assert.assertEquals(0, _producer.getUsedMemory());

		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);
		Assert.assertEquals(1, eos.get());
	}

	@Test
	public void testStoreBackedStreamSubscriber() throws Exception {
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance);
		for(int i = 0; i < 2; i++)
			materializer.accept(new OOCStream.SimpleQueueCallback<>(tile(i, i + 1.0), null));
		materializer.accept(OOCStream.eos(null));
		materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);

		StoreBackedStream<IndexedMatrixValue> stream = new StoreBackedStream<>(
			_store.openReader(new SequentialAccessPattern(2), _readerAllowance, 1));
		_store.sealReaders();
		AtomicInteger count = new AtomicInteger();
		CountDownLatch complete = new CountDownLatch(1);
		stream.setSubscriber(callback -> {
			if(callback.isEos())
				complete.countDown();
			else
				Assert.assertEquals(count.incrementAndGet(), callback.get().getIndexes().getRowIndex());
		});

		Assert.assertTrue(complete.await(WAIT_SECONDS, TimeUnit.SECONDS));
		Assert.assertEquals(2, count.get());
		Assert.assertEquals(0, _readerAllowance.getUsedMemory());
	}

	@Test
	public void testCountingLiveness() {
		CountingLiveness liveness = new CountingLiveness(1, 2);
		Assert.assertTrue(liveness.reserve(0));
		Assert.assertTrue(liveness.reserve(0));
		Assert.assertFalse(liveness.reserve(0));
		liveness.unreserve(0);
		Assert.assertTrue(liveness.reserve(0));
		liveness.consumed(0);
		Assert.assertTrue(liveness.needs(0));
		liveness.consumed(0);
		Assert.assertFalse(liveness.needs(0));
	}

	@Test
	public void testFailurePropagation() throws Exception {
		DMLRuntimeException sourceFailure = new DMLRuntimeException("injected failure");
		AtomicInteger failures = new AtomicInteger();
		OOCStreamMaterializer materializer = new OOCStreamMaterializer(_store,
			indexes -> (int) indexes.getRowIndex() - 1, _materializerAllowance, List.of(callback -> {
				if(callback.isFailure())
					failures.incrementAndGet();
			}));
		materializer.accept(OOCStream.eos(sourceFailure));
		try {
			materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);
			Assert.fail("The source failure must reach materializer completion");
		}
		catch(ExecutionException ex) {
			Assert.assertSame(sourceFailure, ex.getCause());
		}
		Assert.assertEquals(1, failures.get());

		_store.close();
		_store = new MaterializedStore<>(_cache, CachingStream._streamSeq.getNextID());
		_store.close();
		materializer = new OOCStreamMaterializer(_store, indexes -> (int) indexes.getRowIndex() - 1,
			_materializerAllowance);
		_producer.reserveBlocking(TILE_BYTES);
		materializer.accept(new InMemoryQueueCallback(tile(0, 1.0), null, _producer, TILE_BYTES));
		try {
			materializer.completion().get(WAIT_SECONDS, TimeUnit.SECONDS);
			Assert.fail("Publishing into a closed store must fail");
		}
		catch(ExecutionException ex) {
			Assert.assertTrue(ex.getCause() instanceof DMLRuntimeException);
		}
		Assert.assertEquals(0, _producer.getUsedMemory());
	}

	private static IndexedMatrixValue tile(int index, double value) {
		return new IndexedMatrixValue(new MatrixIndexes(index + 1L, 1), new MatrixBlock(4, 4, value));
	}

}
