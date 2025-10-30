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
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Eviction Manager for the Out-Of-Core stream cache
 * This is the base implementation for LRU, FIFO
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
 */
public class OOCEvictionManager {

	// Configuration: OOC buffer limit as percentage of heap
	private static final double OOC_BUFFER_PERCENTAGE = 0.15; // 15% of heap

	// Memory limit for ByteBuffers
	private static long _limit;
	private static AtomicLong _size;

	// Cache of ByteBuffers (off-heap serialized blocks)
	private static CacheEvictionQueue _mQueue;

//	private static Map<Long, Map<Integer, ByteBuffer>> cache = new HashMap<>();
	// I/O service for async spill/load
	private static CacheMaintenanceService _fClean;

	// Spill directory for evicted blocks
	private static String _spillDir;

	public enum RPolicy {
		FIFO, LRU
	}
	private static RPolicy _policy = RPolicy.FIFO;

	private OOCEvictionManager() {}

	static {
		_mQueue = new CacheEvictionQueue();
		_fClean = new CacheMaintenanceService();
		_limit = (long)(Runtime.getRuntime().maxMemory() * OOC_BUFFER_PERCENTAGE * 0.01); // e.g., 20% of heap
		_size = 0;
		_spillDir = LocalFileUtils.getUniqueWorkingDir("ooc_stream");
		LocalFileUtils.createLocalFileIfNotExist(_spillDir);
	}

	/**
	 * Store a block in the OOC cache (serialize once)
	 */
	public static synchronized void put(long streamId, int blockId, IndexedMatrixValue value) {
		try {
			MatrixBlock mb = (MatrixBlock) value.getValue();
			// Serialize to ByteBuffer
			long size = estimateSerializedSize(mb);
			ByteBuffer bbuff = new ByteBuffer(size);

			synchronized (_mQueue) {
				// Make space
				evict(size);

				// Add to cache
				_mQueue.addLast(streamId + "_" + blockId, bbuff);
				_size += size;
			}

			// Serialize outside lock
			_fClean.serializeData(bbuff, mb);
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Get a block from the OOC cache (deserialize on read)
	 */
	public static synchronized IndexedMatrixValue get(long streamId, int blockId) {
		ByteBuffer bbuff = null;
		String key = streamId + "_" + blockId;

		try {
			synchronized (_mQueue) {
				bbuff = _mQueue.get(key);

				// LRU: move to end
				if (_policy == RPolicy.LRU && bbuff != null) {
					_mQueue.remove(key);
					_mQueue.addLast(key, bbuff);
				}
			}

			if (bbuff != null) {
				// Cache hit: deserialize from ByteBuffer
				bbuff.checkSerialized();
				MatrixBlock mb = (MatrixBlock) bbuff.deserializeBlock();

				MatrixIndexes ix = new MatrixIndexes(blockId + 1, 1);
				return new IndexedMatrixValue(ix, mb);
			} else {
				// Cache miss: load from disk
				return loadFromDisk(streamId, blockId);
			}
		}
		catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Evict ByteBuffers to disk
	 */
	private static void evict(long requiredSize) throws IOException {
		while(_size + requiredSize > _limit && !_mQueue.isEmpty()) {
			System.out.println("_size + requiredSize: " + _size +" + "+ requiredSize + "; _limit: " + _limit);
			Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
			String key = entry.getKey();
			ByteBuffer bbuff = entry.getValue();

			if(bbuff != null) {
				// Wait for serialization
				bbuff.checkSerialized();

				// Spill to disk
				String filename = _spillDir + "/" + key;
				File spillDirFile = new File(_spillDir);
				if (!spillDirFile.exists()) {
					spillDirFile.mkdirs();
				}
				System.out.println("Evicting directory: "+ filename);
				bbuff.evictBuffer(filename);
				bbuff.freeMemory();
				_size -= bbuff.getSize();
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