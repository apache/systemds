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

package org.apache.sysds.runtime.controlprogram.caching;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

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
	private static final double OOC_BUFFER_PERCENTAGE = 0.15; // 15% of heap

	// Memory limit for ByteBuffers
	private static long _limit;
	private static long _size;

	// Cache structures: map key -> MatrixBlock and eviction deque (head=oldest block)
	private static final Map<String, MatrixBlock> _cache = new HashMap<>();
	private static final Deque<String> _evictDeque = new ArrayDeque<>();

	// Single lock for synchronization
	private static final Object lock = new Object();

	// Spill directory for evicted blocks
	private static String _spillDir;

	public enum RPolicy {
		FIFO, LRU
	}
	private static RPolicy _policy = RPolicy.FIFO;

	private OOCEvictionManager() {}

	static {
		_limit = (long)(Runtime.getRuntime().maxMemory() * OOC_BUFFER_PERCENTAGE * 0.01); // e.g., 20% of heap
		_size = 0;
		_spillDir = LocalFileUtils.getUniqueWorkingDir("ooc_stream");
		LocalFileUtils.createLocalFileIfNotExist(_spillDir);
	}

	/**
	 * Store a block in the OOC cache (serialize once)
	 */
	public static synchronized void put(long streamId, int blockId, IndexedMatrixValue value) {
		MatrixBlock mb = (MatrixBlock) value.getValue();
		long size = estimateSerializedSize(mb);
		String key = streamId + "_" + blockId;

		synchronized (lock) {
			MatrixBlock old = _cache.remove(key);
			if (old != null) {
				_evictDeque.remove(key);
				_size -= estimateSerializedSize(old);
			}

			try {
				evict(size);
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}

			_cache.put(key, mb);
			_evictDeque.addLast(key); // add to end for FIFO/LRU
			_size += size;
		}
	}

	/**
	 * Get a block from the OOC cache (deserialize on read)
	 */
	public static synchronized IndexedMatrixValue get(long streamId, int blockId) {
		String key = streamId + "_" + blockId;
		MatrixBlock mb = (MatrixBlock) _cache.get(key);

		synchronized (lock) {
			if (mb != null && _policy == RPolicy.LRU) {
				_evictDeque.remove(key);
				_evictDeque.addLast(key);
			}
		}

		if (mb != null) {
			MatrixIndexes ix = new MatrixIndexes(blockId + 1, 1);
			return new IndexedMatrixValue(ix, mb);
		} else {
			try {
				return loadFromDisk(streamId, blockId);
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}

	}

	/**
	 * Evict ByteBuffers to disk
	 */
	private static void evict(long requiredSize) throws IOException {
		while(_size + requiredSize > _limit && !_evictDeque.isEmpty()) {
			System.out.println("_size + requiredSize: " + _size +" + "+ requiredSize + "; _limit: " + _limit);
			String oldKey = _evictDeque.removeLast();
			MatrixBlock mbToEvict = (MatrixBlock) _cache.remove(oldKey);

			if(mbToEvict != null) {

				// Spill to disk
				String filename = _spillDir + "/" + oldKey;
				File spillDirFile = new File(_spillDir);
				if (!spillDirFile.exists()) {
					spillDirFile.mkdirs();
				}

				LocalFileUtils.writeMatrixBlockToLocal(filename, mbToEvict);
				System.out.println("Evicting directory: "+ filename);

				long freedSize = estimateSerializedSize(mbToEvict);
				_size -= freedSize;
			}
		}
	}

	/**
	 * Load block from spill file
	 */
	private static IndexedMatrixValue loadFromDisk(long streamId, int blockId) throws IOException {
		String filename = _spillDir + "/" + streamId + "_" + blockId;

		// check if file exists
		if (!LocalFileUtils.isExisting(filename)) {
			throw new IOException("File " + filename + " does not exist");
		}

		// Read from disk
		MatrixBlock mb = LocalFileUtils.readMatrixBlockFromLocal(filename);

		MatrixIndexes ix = new MatrixIndexes(blockId + 1, 1);

		// Put back in cache (may trigger eviction)
		// get() operation should not modify cache
		// put(streamId, blockId, new IndexedMatrixValue(ix, mb));

		return new IndexedMatrixValue(ix, mb);
	}

	private static long estimateSerializedSize(MatrixBlock mb) {
		return mb.getExactSerializedSize();
	}

}