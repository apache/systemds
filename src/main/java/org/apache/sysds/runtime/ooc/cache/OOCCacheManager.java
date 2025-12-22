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
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
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

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPin(long streamId, int blockId, IndexedMatrixValue value) {
		BlockKey key = new BlockKey(streamId, blockId);
		return new CachedQueueCallback<>(getCache().putAndPin(key, value, ((MatrixBlock)value.getValue()).getExactSerializedSize()), null);
	}

	public static OOCStream.QueueCallback<IndexedMatrixValue> putAndPinSourceBacked(long streamId, int blockId,
		IndexedMatrixValue value, OOCIOHandler.SourceBlockDescriptor descriptor) {
		BlockKey key = new BlockKey(streamId, blockId);
		return new CachedQueueCallback<>(
			getCache().putAndPinSourceBacked(key, value, ((MatrixBlock) value.getValue()).getExactSerializedSize(),
				descriptor), null);
	}

	public static void prioritize(BlockKey key, int priority) {
		getCache().prioritize(key, priority);
	}

	public static CompletableFuture<OOCStream.QueueCallback<IndexedMatrixValue>> requestBlock(long streamId, long blockId) {
		BlockKey key = new BlockKey(streamId, blockId);
		return getCache().request(key).thenApply(e -> new CachedQueueCallback<>(e, null));
	}

	public static CompletableFuture<List<OOCStream.QueueCallback<IndexedMatrixValue>>> requestManyBlocks(List<BlockKey> keys) {
		return getCache().request(keys).thenApply(
			l -> l.stream().map(e -> (OOCStream.QueueCallback<IndexedMatrixValue>)new CachedQueueCallback<IndexedMatrixValue>(e, null)).toList());
	}

	public static List<OOCStream.QueueCallback<IndexedMatrixValue>> tryRequestManyBlocks(List<BlockKey> keys) {
		List<BlockEntry> entries = getCache().tryRequest(keys);
		if(entries == null)
			return null;
		return entries.stream().map(e -> (OOCStream.QueueCallback<IndexedMatrixValue>)new CachedQueueCallback<IndexedMatrixValue>(e, null)).toList();
	}

	public static CompletableFuture<List<OOCStream.QueueCallback<IndexedMatrixValue>>> requestAnyOf(List<BlockKey> keys, int n, List<BlockKey> sel) {
		return getCache().requestAnyOf(keys, n, sel)
			.thenApply(
				l -> l.stream().map(e -> (OOCStream.QueueCallback<IndexedMatrixValue>)new CachedQueueCallback<IndexedMatrixValue>(e, null)).toList());
	}

	public static boolean canClaimMemory() {
		return getCache().isWithinSoftLimits() && OOCInstruction.getComputeInFlight() <= OOCInstruction.getComputeBackpressureThreshold();
	}

	private static void pin(BlockEntry entry) {
		getCache().pin(entry);
	}

	private static void unpin(BlockEntry entry) {
		getCache().unpin(entry);
	}




	static class CachedQueueCallback<T> implements OOCStream.QueueCallback<T> {
		private final BlockEntry _result;
		private final AtomicBoolean _pinned;
		private T _data;
		private DMLRuntimeException _failure;
		private CompletableFuture<Void> _future;

		@SuppressWarnings("unchecked")
		CachedQueueCallback(BlockEntry result, DMLRuntimeException failure) {
			this._result = result;
			this._data = (T)result.getData();
			this._failure = failure;
			this._pinned = new AtomicBoolean(true);
		}

		@SuppressWarnings("unchecked")
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
		public boolean isManagedByCache() {
			return true;
		}

		@Override
		public void close() {
			if(_pinned.compareAndSet(true, false)) {
				_data = null;
				unpin(_result);
				if(_future != null)
					_future.complete(null);
			}
		}
	}
}
