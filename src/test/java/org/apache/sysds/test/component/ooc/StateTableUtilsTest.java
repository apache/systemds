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

import java.util.concurrent.TimeUnit;

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.cache.OOCCacheImpl;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.runtime.ooc.store.MaterializedCallback;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.util.StateTableUtils;
import org.apache.sysds.test.component.ooc.cache.OOCCacheTestUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class StateTableUtilsTest {
	private static final long MEMORY_LIMIT = 100_000_000;
	private static final long WAIT_SECONDS = 10;
	private static final long TILE_BYTES = new MatrixBlock(4, 4, 1.0).getExactSerializedSize();

	private SyncMemoryAllowance _producer;
	private SyncMemoryAllowance _reader;
	private OOCCacheImpl _cache;
	private StateTable<IndexedMatrixValue> _source;
	private StateTable<IndexedMatrixValue> _table;

	@Before
	public void setUp() {
		GlobalMemoryBroker broker = new GlobalMemoryBroker(1_000_000_000);
		_producer = new SyncMemoryAllowance(broker);
		_reader = new SyncMemoryAllowance(broker);
		_producer.setTargetMemory(MEMORY_LIMIT);
		_reader.setTargetMemory(MEMORY_LIMIT);
		_cache = new OOCCacheImpl(new OOCCacheTestUtils.RecordingOOCIOHandler(), MEMORY_LIMIT, MEMORY_LIMIT);
		_source = new StateTable<>(_cache, 1);
		_table = new StateTable<>(_cache, 2);
	}

	@After
	public void tearDown() {
		_source.close();
		_table.close();
		_cache.shutdown();
		_producer.destroy();
		_reader.destroy();
	}

	@Test
	public void testCallbackPutOrTake() throws Exception {
		_producer.reserveBlocking(TILE_BYTES);
		_source.put(0, new ManagedPayload<>(tile(1.0), TILE_BYTES, _producer));
		StoreLease<IndexedMatrixValue> pinned = _source.peek(0, _reader);
		Assert.assertNotNull(pinned);
		Assert.assertNull(StateTableUtils.putOrTake(_table, 0, new MaterializedCallback(pinned), _reader)
			.get(WAIT_SECONDS, TimeUnit.SECONDS));
		Assert.assertEquals(0, _reader.getUsedMemory());

		_producer.reserveBlocking(TILE_BYTES);
		StateTableUtils.Match referenced = StateTableUtils
			.putOrTake(_table, 0, new InMemoryQueueCallback(tile(2.0), null, _producer, TILE_BYTES), _reader)
			.get(WAIT_SECONDS, TimeUnit.SECONDS);
		Assert.assertNotNull(referenced);
		try(OOCStream.QueueCallback<IndexedMatrixValue> left = referenced.left();
			OOCStream.QueueCallback<IndexedMatrixValue> right = referenced.right()) {
			Assert.assertEquals(2.0, left.get().getValue().get(0, 0), 0.0);
			Assert.assertEquals(1.0, right.get().getValue().get(0, 0), 0.0);
		}

		_producer.reserveBlocking(TILE_BYTES);
		Assert.assertNull(StateTableUtils
			.putOrTake(_table, 1, new InMemoryQueueCallback(tile(3.0), null, _producer, TILE_BYTES), _reader)
			.get(WAIT_SECONDS, TimeUnit.SECONDS));
		StateTableUtils.Match copied = StateTableUtils
			.putOrTake(_table, 1, new OOCStream.SimpleQueueCallback<>(tile(4.0), null), _reader)
			.get(WAIT_SECONDS, TimeUnit.SECONDS);
		Assert.assertNotNull(copied);
		try(OOCStream.QueueCallback<IndexedMatrixValue> own = copied.left();
			OOCStream.QueueCallback<IndexedMatrixValue> partner = copied.right()) {
			Assert.assertEquals(4.0, own.get().getValue().get(0, 0), 0.0);
			Assert.assertEquals(3.0, partner.get().getValue().get(0, 0), 0.0);
		}

		Assert.assertEquals(0, _producer.getUsedMemory());
		Assert.assertEquals(0, _reader.getUsedMemory());
		_source.close();
		_table.close();
		OOCCacheTestUtils.await(() -> _cache.getOwnedCacheSize() == 0, WAIT_SECONDS);
	}

	@Test
	public void testStateTableLifecycle() throws Exception {
		_producer.reserveBlocking(TILE_BYTES);
		_table.put(0, new ManagedPayload<>(tile(5.0), TILE_BYTES, _producer));
		try(StoreLease<IndexedMatrixValue> lease = _table.acquire(0, _reader).get(WAIT_SECONDS, TimeUnit.SECONDS)) {
			Assert.assertNotNull(lease);
			Assert.assertEquals(5.0, lease.value().getValue().get(0, 0), 0.0);
		}
		try(StoreLease<IndexedMatrixValue> lease = _table.take(0, _reader).get(WAIT_SECONDS, TimeUnit.SECONDS)) {
			Assert.assertNotNull(lease);
			Assert.assertEquals(5.0, lease.value().getValue().get(0, 0), 0.0);
		}
		Assert.assertNull(_table.take(0, _reader).get(WAIT_SECONDS, TimeUnit.SECONDS));

		_producer.reserveBlocking(TILE_BYTES);
		_table.put(1, new ManagedPayload<>(tile(6.0), TILE_BYTES, _producer));
		_table.clear(1);
		Assert.assertNull(_table.take(1, _reader).get(WAIT_SECONDS, TimeUnit.SECONDS));
		Assert.assertEquals(0, _producer.getUsedMemory());
		Assert.assertEquals(0, _reader.getUsedMemory());
		OOCCacheTestUtils.await(() -> _cache.getOwnedCacheSize() == 0, WAIT_SECONDS);
	}

	private static IndexedMatrixValue tile(double value) {
		return new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(4, 4, value));
	}
}
