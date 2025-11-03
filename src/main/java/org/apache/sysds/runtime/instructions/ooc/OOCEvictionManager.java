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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Eviction Manager for the Out-Of-Core stream cache
 * This is the base implementation for LRU, FIFO
 *
 * Design choice 1: Pure JVM-memory cache
 * What: Store MatrixBlock objects in a synchronized in-memory cache
 *   (Map + Deque for LRU/FIFO). Spill to disk by serializing MatrixBlock
 *   only when evicting.
 * Pros: Simple to implement; no off-heap management; easy to debug;
 *   no serialization race since you serialize only when evicting;
 *   fast cache hits (direct object access).
 * Cons: Heap usage counted roughly via serialized-size estimate — actual
 *   JVM object overhead not accounted; risk of GC pressure and OOM if
 *   estimates are off or if many small objects cause fragmentation;
 *   eviction may be more expensive (serialize on eviction).
 * <p>
 * Design choice 2:
 * <p>
 * This manager runtime memory management by caching serialized
 * ByteBuffers and spilling them to disk when needed.
 * <p>
 * * core function: Caches ByteBuffers (off-heap/direct) and
 * spills them to disk
 * * Eviction: Evicts a ByteBuffer by writing its contents to a file
 * * Granularity: Evicts one IndexedMatrixValue block at a time
 * * Data replay: get() will always return the data either from memory or
 *   by falling back to the disk
 * * Memory: Since the datablocks are off-heap (in ByteBuffer) or disk,
 *   there won't be OOM.
 *
 * Pros: Avoids heap OOM by keeping large data off-heap; predictable
 *   memory usage; good for very large blocks.
 * Cons: More complex synchronization; need robust off-heap allocator/free;
 *   must ensure serialization finishes before adding to queue or make evict
 *   wait on serialization; careful with native memory leaks.
 */
public class OOCEvictionManager {

	// Configuration: OOC buffer limit as percentage of heap
	private static final double OOC_BUFFER_PERCENTAGE = 0.15 * 0.01; // 15% of heap

	// Memory limit for ByteBuffers
	private static long _limit;
	private static final AtomicLong _size = new AtomicLong(0);

	// Cache structures: map key -> MatrixBlock and eviction deque (head=oldest block)
	private static LinkedHashMap<String, BlockEntry> _cache = new LinkedHashMap<>();

	// Cache level lock
	private static final Object _cacheLock = new Object();
	
	// Spill directory for evicted blocks
	private static String _spillDir;

	public enum RPolicy {
		FIFO, LRU
	}
	private static RPolicy _policy = RPolicy.FIFO;

	private enum BlockState {
		HOT, // In-memory
		EVICTING, // Being written to disk (transition state)
		COLD // On disk
	}

	// Per-block state container with own lock.
	private static class BlockEntry {
		private final ReentrantLock lock = new ReentrantLock();
		private final Condition stateUpdate = lock.newCondition();

		private BlockState state = BlockState.HOT;
		private IndexedMatrixValue value;
		private final long streamId;
		private final int blockId;
		private final long size;

		BlockEntry(IndexedMatrixValue value, long streamId, int blockId, long size) {
			this.value = value;
			this.streamId = streamId;
			this.blockId = blockId;
			this.size = size;
		}
	}

	static {
		_limit = (long)(Runtime.getRuntime().maxMemory() * OOC_BUFFER_PERCENTAGE); // e.g., 20% of heap
		_size.set(0);
		_spillDir = LocalFileUtils.getUniqueWorkingDir("ooc_stream");
		LocalFileUtils.createLocalFileIfNotExist(_spillDir);
	}

	/**
	 * Store a block in the OOC cache (serialize once)
	 */
	public static void put(long streamId, int blockId, IndexedMatrixValue value) {
		MatrixBlock mb = (MatrixBlock) value.getValue();
		long size = estimateSerializedSize(mb);
		String key = streamId + "_" + blockId;

		BlockEntry newEntry = new BlockEntry(value, streamId, blockId, size);
		BlockEntry old;
		synchronized (_cacheLock) {
			old = _cache.put(key, newEntry); // remove old value, put new value
		}

		// Handle replacement with a new lock
		if (old != null) {
			old.lock.lock();
			try {
				if (old.state == BlockState.HOT) {
					_size.addAndGet(-old.size); // read and update size in atomic operation
				}
			} finally {
				old.lock.unlock();
			}
		}

		_size.addAndGet(size);
		//make room if needed
		evict(size);
	}

	/**
	 * Get a block from the OOC cache (deserialize on read)
	 */
	public static IndexedMatrixValue get(long streamId, int blockId) {
		String key = streamId + "_" + blockId;
		BlockEntry imv;

		synchronized (_cacheLock) {
			imv = _cache.get(key);

			if (imv != null && _policy == RPolicy.LRU) {
				_cache.remove(key);
				_cache.put(key, imv); //add last semantic
			}
		}

		// use lock and check state
		imv.lock.lock();
		try {
			// 1. wait for eviction to complete
			while (imv.state == BlockState.EVICTING) {
				try {
					imv.stateUpdate.wait();
				} catch (InterruptedException e) {

					throw new DMLRuntimeException(e);
				}
			}

			// 2. check if the block is in HOT
			if (imv.state == BlockState.HOT) {
				return imv.value;
			}

		} finally {
			imv.lock.unlock();
		}

		// restore, since the block is COLD
		return loadFromDisk(streamId, blockId);
	}

	/**
	 * Evict ByteBuffers to disk
	 */
	private static void evict(long requiredSize) {
		try {
			int pos = 0;
			while(_size.get() > _limit && pos++ < _cache.size()) {
//				System.out.println("BUFFER: "+_size+"/"+_limit+" size="+_cache.size());
				Map.Entry<String,BlockEntry> tmp = removeFirstFromCache();
				if( tmp == null || tmp.getValue().value.getValue() == null ) {
					if( tmp != null )
						_cache.put(tmp.getKey(), tmp.getValue());
					continue;
				}
	
				// Spill to disk
				String filename = _spillDir + "/" + tmp.getKey();
				File spillDirFile = new File(_spillDir);
				if (!spillDirFile.exists()) {
					spillDirFile.mkdirs();
				}
				LocalFileUtils.writeMatrixBlockToLocal(filename, (MatrixBlock)tmp.getValue().value.getValue());
	
				// Evict from memory
				long freedSize = estimateSerializedSize((MatrixBlock)tmp.getValue().value.getValue());

				BlockEntry entry = tmp.getValue();

				entry.lock.lock();
				try {
					entry.value.setValue(null);
					entry.state = BlockState.COLD; // set state to cold, since writing to disk


				} finally {
					entry.lock.unlock();
				}

				_cache.put(tmp.getKey(), entry); // add last semantic
				_size.addAndGet(-freedSize);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	/**
	 * Load block from spill file
	 */
	private static IndexedMatrixValue loadFromDisk(long streamId, int blockId) {
		String key = streamId + "_" + blockId;
		String filename = _spillDir + "/" + key;

		try {
			// check if file exists
			if (!LocalFileUtils.isExisting(filename)) {
				throw new IOException("File " + filename + " does not exist");
			}
	
			// Read from disk and put into original indexed matrix value
			MatrixBlock mb = LocalFileUtils.readMatrixBlockFromLocal(filename);
			BlockEntry imv = _cache.get(key);
			imv.value.setValue(mb);
			return imv.value;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private static long estimateSerializedSize(MatrixBlock mb) {
		return mb.getExactSerializedSize();
	}
	
	private static Map.Entry<String, BlockEntry> removeFirstFromCache() {
		//move iterator to first entry
		Iterator<Map.Entry<String, BlockEntry>> iter = _cache.entrySet().iterator();
		Map.Entry<String, BlockEntry> entry = iter.next();

		//remove current iterator entry
		iter.remove();

		return entry;
	}
}
