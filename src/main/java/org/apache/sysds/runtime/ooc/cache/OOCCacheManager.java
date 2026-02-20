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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCInstruction;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.utils.Statistics;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class OOCCacheManager {
	private static final double OOC_BUFFER_PERCENTAGE = 0.2;
	private static final double OOC_BUFFER_PERCENTAGE_HARD = 0.3;
	private static final long _evictionLimit;
	private static final long _hardLimit;

	private static final AtomicReference<OOCIOHandler> _ioHandler;
	private static final AtomicReference<OOCCacheScheduler> _scheduler;

	static {
		_evictionLimit = (long)(Runtime.getRuntime().maxMemory() * OOC_BUFFER_PERCENTAGE);
		_hardLimit = (long)(Runtime.getRuntime().maxMemory() * OOC_BUFFER_PERCENTAGE_HARD);
		_ioHandler = new AtomicReference<>();
		_scheduler = new AtomicReference<>();
	}

	public static void reset() {
		OOCIOHandler ioHandler = _ioHandler.getAndSet(null);
		OOCCacheScheduler cacheScheduler = _scheduler.getAndSet(null);
		if (ioHandler != null)
			ioHandler.shutdown();
		if (cacheScheduler != null)
			cacheScheduler.shutdown();

		if (DMLScript.OOC_STATISTICS)
			Statistics.resetOOCEvictionStats();

		if (DMLScript.OOC_LOG_EVENTS) {
			try {
				String csv = OOCEventLog.getComputeEventsCSV();
				Files.writeString(Path.of(DMLScript.OOC_LOG_PATH, "ComputeEventLog.csv"), csv);
				csv = OOCEventLog.getDiskReadEventsCSV();
				Files.writeString(Path.of(DMLScript.OOC_LOG_PATH, "DiskReadEventLog.csv"), csv);
				csv = OOCEventLog.getDiskWriteEventsCSV();
				Files.writeString(Path.of(DMLScript.OOC_LOG_PATH, "DiskWriteEventLog.csv"), csv);
				csv = OOCEventLog.getCacheSizeEventsCSV();
				Files.writeString(Path.of(DMLScript.OOC_LOG_PATH, "CacheSizeEventLog.csv"), csv);
				csv = OOCEventLog.getRunSettingsCSV();
				Files.writeString(Path.of(DMLScript.OOC_LOG_PATH, "RunSettings.csv"), csv);
				System.out.println("Event logs written to: " + DMLScript.OOC_LOG_PATH);
			}
			catch(IOException e) {
				System.err.println("Could not write event logs: " + e.getMessage());
			}
			OOCEventLog.clear();
		}
	}

	public static OOCCacheScheduler getCache() {
		while (true) {
			OOCCacheScheduler scheduler = _scheduler.get();

			if(scheduler != null)
				return scheduler;

			OOCIOHandler ioHandler = new OOCMatrixIOHandler();
			scheduler = new OOCLRUCacheScheduler(ioHandler, _evictionLimit, _hardLimit);

			if(_scheduler.compareAndSet(null, scheduler)) {
				_ioHandler.set(ioHandler);
				return scheduler;
			}
		}
	}

	public static OOCIOHandler getIOHandler() {
		OOCIOHandler io = _ioHandler.get();
		if(io != null)
			return io;
		// Ensure initialization happens
		getCache();
		return _ioHandler.get();
	}

	/**
	 * Removes a block from the cache without setting its data to null.
	 */
	public static void forget(long streamId, int blockId) {
		BlockKey key = new BlockKey(streamId, blockId);
		getCache().forget(key);
	}

	/**
	 * Store a block in the OOC cache (serialize once)
	 */
	public static void put(long streamId, int blockId, IndexedMatrixValue value) {
		BlockKey key = new BlockKey(streamId, blockId);
		getCache().put(key, value, ((MatrixBlock)value.getValue()).getExactSerializedSize());
	}

	/**
	 * Store a source-backed block in the OOC cache and register its source location.
	 */
	public static void putSourceBacked(long streamId, int blockId, IndexedMatrixValue value,
		OOCIOHandler.SourceBlockDescriptor descriptor) {
		BlockKey key = new BlockKey(streamId, blockId);
		getCache().putSourceBacked(key, value, ((MatrixBlock) value.getValue()).getExactSerializedSize(), descriptor);
	}

	public static void putRawSourceBacked(BlockKey key, Object data, long size, OOCIOHandler.SourceBlockDescriptor descriptor) {
		getCache().putSourceBacked(key, data, size, descriptor);
	}

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPin(long streamId, int blockId, IndexedMatrixValue value) {
		BlockKey key = new BlockKey(streamId, blockId);
		return new CachedQueueCallback<>(getCache().putAndPin(key, value, ((MatrixBlock)value.getValue()).getExactSerializedSize()), null);
	}

	public static void putRaw(BlockKey key, Object data, long size) {
		getCache().put(key, data, size);
	}

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPinRaw(BlockKey key, Object data, long size) {
		BlockEntry entry = getCache().putAndPin(key, data, size);
		if (data instanceof List)
			return new CachedGroupCallback<>(entry, null);
		return new CachedQueueCallback<>(entry, null);
	}

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPinSourceBacked(long streamId, int blockId,
		IndexedMatrixValue value, OOCIOHandler.SourceBlockDescriptor descriptor) {
		BlockKey key = new BlockKey(streamId, blockId);
		return new CachedQueueCallback<>(
			getCache().putAndPinSourceBacked(key, value, ((MatrixBlock) value.getValue()).getExactSerializedSize(),
				descriptor), null);
	}

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPinRawSourceBacked(BlockKey key, Object data, long size,
		OOCIOHandler.SourceBlockDescriptor descriptor) {
		BlockEntry entry = getCache().putAndPinSourceBacked(key, data, size, descriptor);
		if (data instanceof List)
			return new CachedGroupCallback<>(entry, null);
		return new CachedQueueCallback<>(entry, null);
	}

	public static void prioritize(BlockKey key, int priority) {
		getCache().prioritize(key, priority);
	}

	public static CompletableFuture<OOCStream.QueueCallback<IndexedMatrixValue>> requestBlock(long streamId, long blockId) {
		return requestBlock(new BlockKey(streamId, (int)blockId));
	}

	public static CompletableFuture<OOCStream.QueueCallback<IndexedMatrixValue>> requestBlock(BlockKey key) {
		return getCache().request(key).thenApply(e -> toCallback(e, key, null));
	}

	public static CompletableFuture<List<OOCStream.QueueCallback<IndexedMatrixValue>>> requestManyBlocks(List<BlockKey> keys) {
		return getCache().request(keys).thenApply(
			l -> {
				List<OOCStream.QueueCallback<IndexedMatrixValue>> out = new ArrayList<>(l.size());
				for (int i = 0; i < l.size(); i++)
					out.add(toCallback(l.get(i), keys.get(i), null));
				return out;
			});
	}

	public static List<OOCStream.QueueCallback<IndexedMatrixValue>> tryRequestManyBlocks(List<BlockKey> keys) {
		List<BlockEntry> entries = getCache().tryRequest(keys);
		if(entries == null)
			return null;
		List<OOCStream.QueueCallback<IndexedMatrixValue>> out = new ArrayList<>(entries.size());
		for (int i = 0; i < entries.size(); i++)
			out.add(toCallback(entries.get(i), keys.get(i), null));
		return out;
	}

	public static CompletableFuture<List<OOCStream.QueueCallback<IndexedMatrixValue>>> requestAnyOf(List<BlockKey> keys, int n, List<BlockKey> sel) {
		return getCache().requestAnyOf(keys, n, sel)
			.thenApply(
				l -> {
					List<OOCStream.QueueCallback<IndexedMatrixValue>> out = new ArrayList<>(l.size());
					for (int i = 0; i < l.size(); i++) {
						BlockKey key = sel.size() == l.size() ? sel.get(i) : keys.get(i);
						out.add(toCallback(l.get(i), key, null));
					}
					return out;
				});
	}

	private static OOCStream.QueueCallback<IndexedMatrixValue> toCallback(BlockEntry entry, BlockKey key, DMLRuntimeException failure) {
		if (entry.getData() instanceof List<?>) {
			CachedGroupCallback<IndexedMatrixValue> group = new CachedGroupCallback<>(entry, failure);
			if (key instanceof GroupedBlockKey gk) {
				OOCStream.QueueCallback<IndexedMatrixValue> sub = group.getCallback(gk.getGroupIndex());
				group.close(); // drop the group-level pin, sub keeps it pinned
				return sub;
			}
			return group;
		}
		return new CachedQueueCallback<>(entry, failure);
	}

	public static boolean canClaimMemory() {
		return getCache().isWithinLimits() && OOCInstruction.getComputeInFlight() <= OOCInstruction.getComputeBackpressureThreshold();
	}

	private static void pin(BlockEntry entry) {
		getCache().pin(entry);
	}

	private static void unpin(BlockEntry entry) {
		getCache().unpin(entry);
	}




	public static class CachedQueueCallback<T> implements OOCStream.QueueCallback<T> {
		private final BlockEntry _result;
		private final AtomicBoolean _pinned;
		private T _data;
		private DMLRuntimeException _failure;

		@SuppressWarnings("unchecked")
		CachedQueueCallback(BlockEntry result, DMLRuntimeException failure) {
			this._result = result;
			this._data = (T)result.getData();
			this._failure = failure;
			this._pinned = new AtomicBoolean(true);
		}

		@Override
		public T get() {
			if(_failure != null)
				throw _failure;
			if(!_pinned.get())
				throw new IllegalStateException("Cannot get cached item of a closed callback");
			return _data;
		}

		@Override
		public OOCStream.QueueCallback<T> keepOpen() {
			if(!_pinned.get())
				throw new IllegalStateException("Cannot keep open an already closed callback");
			pin(_result);
			return new CachedQueueCallback<>(_result, _failure);
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			this._failure = failure;
		}

		@Override
		public boolean isEos() {
			return get() == null;
		}

		@Override
		public boolean isFailure() {
			return _failure != null;
		}

		@Override
		public void close() {
			if(_pinned.compareAndSet(true, false)) {
				_data = null;
				unpin(_result);
			}
		}

		public BlockKey getBlockKey() {
			return _result.getKey();
		}
	}

	public static class CachedSubCallback<T> implements OOCStream.QueueCallback<T> {
		private final CachedGroupCallback<T> _parent;
		private final AtomicBoolean _pinned;
		private T _data;
		private final int _groupIndex;

		CachedSubCallback(CachedGroupCallback<T> parent, T data, int groupIndex) {
			_parent = parent;
			_data = data;
			_groupIndex = groupIndex;
			_pinned = new AtomicBoolean(true);
		}

		@Override
		public T get() {
			if(_parent.isFailure())
				throw _parent._failure;
			return _data;
		}

		@Override
		public OOCStream.QueueCallback<T> keepOpen() {
			_parent.registerQueueCallback();
			return new CachedSubCallback<>(_parent, _data, _groupIndex);
		}

		@Override
		public void close() {
			if(_pinned.compareAndSet(true, false)) {
				_data = null;
				_parent.close();
			}
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			_parent.fail(failure);
		}

		@Override
		public boolean isEos() {
			return false;
		}

		@Override
		public boolean isFailure() {
			return _parent.isFailure();
		}

		public CachedGroupCallback<T> getParent() {
			return _parent;
		}

		public int getGroupIndex() {
			return _groupIndex;
		}
	}

	public static class CachedGroupCallback<T> implements OOCStream.GroupQueueCallback<T> {
		private final BlockEntry _result;
		private final AtomicInteger _pinCounter;
		private List<T> _data;
		private DMLRuntimeException _failure;

		@SuppressWarnings("unchecked")
		CachedGroupCallback(BlockEntry result, DMLRuntimeException failure) {
			this._result = result;
			this._data = (List<T>)result.getData();
			this._failure = failure;
			this._pinCounter = new AtomicInteger(1);
		}

		public OOCStream.QueueCallback<T> getCallback(int idx) {
			if(_pinCounter.get() <= 0)
				throw new IllegalStateException("Cannot open sub-callback on a closed GroupCallback");
			registerQueueCallback();
			return new CachedSubCallback<>(this, _data.get(idx), idx);
		}

		public void registerQueueCallback() {
			if(_pinCounter.incrementAndGet() <= 1)
				throw new IllegalStateException();
		}

		@Override
		public T get() {
			throw new UnsupportedOperationException();
		}

		@Override
		public int size() {
			return _data.size();
		}

		public T get(int idx) {
			return _data.get(idx);
		}

		@Override
		public OOCStream.QueueCallback<T> keepOpen() {
			if(_pinCounter.get() <= 0)
				throw new IllegalStateException("Cannot keep open an already closed callback");
			pin(_result);
			return new CachedGroupCallback<>(_result, _failure);
		}

		@Override
		public void close() {
			int cnt = _pinCounter.decrementAndGet();
			if(cnt == 0) {
				_data = null;
				unpin(_result);
			}
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			_failure = failure;
		}

		@Override
		public boolean isEos() {
			return false;
		}

		@Override
		public boolean isFailure() {
			return _failure != null;
		}

		public BlockKey getBlockKey() {
			return _result.getKey();
		}
	}
}
