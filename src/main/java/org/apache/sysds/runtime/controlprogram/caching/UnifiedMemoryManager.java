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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Unified Memory Manager - Initial Design
 * 
 * Motivation:
 * The Unified Memory Manager, henceforth UMM, will act as a central manager of in-memory
 * matrix (uncompressed and compressed), frame, and tensor blocks within SystemDS control
 * program. So far, operation memory (70%) and buffer pool memory (15%, LazyWriteBuffer)
 * are managed independently, which causes unnecessary evictions. New components like the
 * LineageCache also use and manage statically provisioned memory areas. Ultimately, the
 * UMM aims to eliminate these shortcomings by providing a central, potentially thread-local,
 * memory management.
 *
 * Memory Areas:
 * Initially, the UMM only handles CacheBlock objects (e.g., MatrixBlock, FrameBlock, and
 * TensorBlock), and manages two memory areas:
 *   (1) operation memory (pinned cache blocks and reserved memory) and
 *   (2) dirty objects (dirty cache blocks that need to be written to local FS before eviction)
 * 
 * The UMM is configured with a capacity (absolute size in byte). Relative to this capacity,
 * the operations and buffer pool memory areas each will have a min and max amount of memory
 * they can occupy, meaning that the boundary for the areas can shift dynamically depending
 * on the current load. Most importantly, though, dirty objects must not be counted twice
 * when pinning such an object for an operation. The min/max constraints are not exposed but
 * configured internally. An good starting point are the following constraints (relative to
 * JVM max heap size):
 * ___________________________
 * | operations  | 0%  | 70% | (pin requests always accepted)
 * | buffer pool | 15% | 85% | (eviction on demand)
 *
 * Object Lifecycle:
 * The UMM will also need to keep track of the current state of individual cache blocks, for
 * which it will have a few member variables. A queue similar to the current EvictionQueue is
 * used to add/remove entries with LRU as its eviction policy. In general, there are three
 * properties of object status to consider:
 *  (1) Non-dirty/dirty: non-dirty objects have a representation on HDFS or can be recomputed
 *      from lineage trace (e.g., rand/seq outputs), while dirty objects need to be preserved.
 *  (2) FS Persisted: on eviction, dirty objects need to be written to local file system.
 *      As long the local file representation exist, dirty objects can simply be dropped.
 *  (3) Pinned/unpinned: For operations, objects are pinned into memory to guard against
 *      eviction. All pin requests have to be accepted, and once a non-dirty object is released
 *      (unpinned) it can be dropped without persisting it to local FS.
 *
 * Example Scenarios for an Operation:
 *  (1) Inputs are available in the UMM, enough space left for the output.
 *  (2) Some inputs are pre-evicted. Read and pin those in the operational memory.
 *  (3) Inputs are available in the UMM, not enough space left for the output.
 *  	Evict cached objects to reserve worst-case output memory.
 *  (4) Some inputs are pre-evicted and not enough space left for the inputs
 *  	and output. Evict cache objects to make space for the inputs.
 *  	Evict cached objects to reserve worst-case output memory.
 *
 * Thread-safeness:
 * Initially, the UMM will be used in an instance-based manner. For global visibility and
 * use in parallel for loops, the UMM would need to provide a static, synchronized API, but
 * this constitutes a source of severe contention. In the future, we will consider a design
 * with thread-local UMMs for the individual parfor workers.
 */

public class UnifiedMemoryManager
{
	// Maximum size of UMM in bytes (85%)
	private static long _limit;
	// Current size in bytes
	private static long _size;
	// Operational memory limit in bytes
	private static long _opMemLimit;
	// List of pinned entries
	private static final List<String> _pinnedEntries = new ArrayList<String>();
	// List of entries read from soft references/rdd/fed/gpu
	private static final List<String> _readEntries = new ArrayList<String>();

	// Eviction queue of <filename,buffer> pairs (implemented via linked hash map
	// for (1) queue semantics and (2) constant time get/insert/delete operations)
	private static CacheEvictionQueue _mQueue;

	// Maintenance service for synchronous or asynchronous delete of evicted files
	private static CacheMaintenanceService _fClean;

	// Pinned size of physical memory. Starts from 0 for an operation. Max is 70% of heap
	// This increases only if the input is not present in the cache and read from FS
	private static long _pinnedPhysicalMemSize = 0;
	// Size of pinned virtual memory. This tracks the total input size
	// This increases if the input is available in the cache.
	private static long _pinnedVirtualMemSize = 0;

	// Pins a cache block into operation memory.
	/*public static void pin(CacheableData<?> cd) {
		if (!CacheableData.isCachingActive()) {
			cd.acquire(false, !cd.isBlobPresent());
			return;
		}

		long estimatedSize = OptimizerUtils.estimateSize(cd.getDataCharacteristics());
		// Entries read from soft reference/rdd/fed/gpu take physical space in the operation memory
		if (cd.isBlobPresent()) {
			// Maintain cache status
			cd.acquire(false, false);
			long toPinSize = cd.getDataSize();
			makeSpace(toPinSize); //TODO: make space before reading from rdd/fed/gpu
			_pinnedPhysicalMemSize += toPinSize;
			_readEntries.add(cd.getCacheFilePathAndName());
		}
		else if (probe(cd)) {
			// Read from cache to operation memory and pin
			cd.acquire(false, true);
			long toPinSize = cd.getDataSize();
			// TODO: Consider deserialization overhead for sparse matrices
			_pinnedVirtualMemSize += toPinSize;
		}
		else {
			// Make space for the estimated size after pinning
			makeSpace(estimatedSize);
			// Restore from FS to operation memory and pin
			cd.acquire(false, true);
			long toPinSize = cd.getDataSize(); //actual size
			_pinnedPhysicalMemSize += toPinSize;
		}
		_pinnedEntries.add(cd.getCacheFilePathAndName());

		// Reserve space for output after pinning every input.
		// This overly conservative approach removes the need to call reserveOutputMem() from
		// each instruction. Ideally, every instruction first pins all the inputs, followed
		// by reserving space for output.
		reserveOutputMem();
	}*/

	public static void pin(CacheableData<?> cd) {
		if (!CacheableData.isCachingActive()) {
			cd.acquire(false, !cd.isBlobPresent());
			return;
		}

		long estimatedSize = OptimizerUtils.estimateSize(cd.getDataCharacteristics());
		if (probe(cd))
			_pinnedVirtualMemSize += estimatedSize;
		else {
			makeSpace(estimatedSize);
			_pinnedPhysicalMemSize += estimatedSize;
		}
		_pinnedEntries.add(cd.getCacheFilePathAndName());
		reserveOutputMem();
	}

	// Reserve space for output in the operation memory
	public static void reserveOutputMem() {
		if (!OptimizerUtils.isUMMEnabled() || !CacheableData.isCachingActive())
			return;
		// Worst case upper bound for output = 70% - size(inputs)
		// FIXME: Parfor splits this 70% into smaller limits
		// TODO: Estimate output size from the input sizes. Reserve only that much.
		long maxOutputSize = _opMemLimit - (_pinnedVirtualMemSize + _pinnedPhysicalMemSize);
		// Evict cached entries to make space in operation memory if needed
		makeSpace(maxOutputSize);

		// Reserve max output memory
		//_pinnedVirtualMemSize += maxOutSize;
	}
	
	 // Unpins (releases) a cache block from operation memory. If the size of
	 // the provided cache block differs from the UMM meta data, the UMM meta
	 // data is updated. Use cases include update-in-place operations and
	 // size reservations via worst-case upper bound estimates.
	/*public static void unpin(CacheableData<?> cd) {
		if (CacheableData.isCachingActive())
			return;

		// TODO: Track preserved output memory to protect from other threads
		if (!_pinnedEntries.contains(cd.getCacheFilePathAndName()))
			return; //unpinned. output of an instruction
		long toUnpinSize = cd.getDataSize();
		// Update total pinned memory size
		if (_readEntries.contains(cd.getCacheFilePathAndName())) {
			_pinnedPhysicalMemSize -= toUnpinSize;
			_readEntries.remove(cd.getCacheFilePathAndName());
			return;
		}
		if (probe(cd))
			_pinnedVirtualMemSize -= toUnpinSize;
		else
			_pinnedPhysicalMemSize -= toUnpinSize;

		_pinnedEntries.remove(cd.getCacheFilePathAndName());
	}*/

	public static void unpin(CacheableData<?> cd) {
		if (!CacheableData.isCachingActive())
			return;

		if (!_pinnedEntries.contains(cd.getCacheFilePathAndName()))
			return; //unpinned. output of an instruction
		long estimatedSize = OptimizerUtils.estimateSize(cd.getDataCharacteristics());
		if (probe(cd))
			_pinnedVirtualMemSize -= estimatedSize;
		else
			_pinnedPhysicalMemSize -= estimatedSize;

		_pinnedEntries.remove(cd.getCacheFilePathAndName());
	}
	
	/**
	 * Removes all cache blocks from all memory areas and deletes all evicted
	 * representations (files in local FS). All internally thread pools must be
	 * shut down in a gracefully manner (e.g., wait for pending deletes).
	 */
	public void deleteAll() {
		//TODO implement
		throw new NotImplementedException();
	}

	public static void init() {
		_mQueue = new CacheEvictionQueue();
		_fClean = new CacheMaintenanceService();
		_limit = OptimizerUtils.getBufferPoolLimit();
		_opMemLimit = (long)(OptimizerUtils.getLocalMemBudget()); //70% of heap
		_size = 0;
		_pinnedPhysicalMemSize = 0;
		_pinnedVirtualMemSize = 0;
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.init();
	}

	public static void cleanup() {
		if( _mQueue != null )
			_mQueue.clear();
		if( _fClean != null )
			_fClean.close();
		if( CacheableData.CACHING_BUFFER_PAGECACHE )
			PageCache.clear();
		_size = 0;
		_pinnedPhysicalMemSize = 0;
		_pinnedVirtualMemSize = 0;
	}

	public static void setUMMLimit(long val) {
		_limit = val;
	}

	public static long getUMMFree() {
		synchronized(_mQueue) {
			return _limit - (_size + _pinnedPhysicalMemSize);
		}
	}

	public static long getUMMSize() {
		synchronized(_mQueue) {
			return _limit;
		}
	}

	public static CacheBlock readBlock(String fname, boolean matrix)
		throws IOException
	{
		CacheBlock cb = null;
		ByteBuffer ldata = null;

		//probe write buffer
		synchronized (_mQueue)
		{
			ldata = _mQueue.get(fname);

			//modify eviction order (accordingly to access)
			if (CacheableData.CACHING_BUFFER_POLICY == LazyWriteBuffer.RPolicy.LRU
				&& ldata != null)
			{
				//reinsert entry at end of eviction queue
				_mQueue.remove (fname);
				_mQueue.addLast (fname, ldata);
			}
		}

		//deserialize or read from FS if required
		if( ldata != null )
		{
			cb = ldata.deserializeBlock();
			if (DMLScript.STATISTICS)
				CacheStatistics.incrementFSBuffHits();
		}
		else
		{
			cb = LocalFileUtils.readCacheBlockFromLocal(fname, matrix);
			if (DMLScript.STATISTICS)
				CacheStatistics.incrementFSHits();
		}

		return cb;
	}

	public static boolean probe(CacheableData<?> cd) {
		String filePath = cd.getCacheFilePathAndName();
		return _mQueue.containsKey(filePath);
	}

	public static void makeSpace(long reqSpace) {
		// Check if sufficient space is already available
		if (getUMMFree() > reqSpace)
			return;

		// Evict cached objects to make space
		int numEvicted = 0;
		try {
			synchronized(_mQueue) {
				// Evict blobs to make room (by default FIFO)
				while (getUMMFree() < reqSpace && !_mQueue.isEmpty()) {
					//remove first unpinned entry from eviction queue
					var entry = _mQueue.removeFirstUnpinned(_pinnedEntries);
					String ftmp = entry.getKey();
					ByteBuffer tmp = entry.getValue();

					if(tmp != null) {
						//wait for pending serialization
						tmp.checkSerialized();
						//evict matrix
						tmp.evictBuffer(ftmp);
						tmp.freeMemory();
						_size -= tmp.getSize();
						numEvicted++;
					}
				}
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Eviction request of size "+(reqSpace-getUMMFree())+ " in the UMM failed.", e);
		}

		if( DMLScript.STATISTICS ) {
			CacheStatistics.incrementFSBuffWrites();
			CacheStatistics.incrementFSWrites(numEvicted);
		}
	}

	public static int writeBlock(String fname, CacheBlock cb)
		throws IOException
	{
		//obtain basic meta data of cache block
		long lSize = getCacheBlockSize(cb);
		boolean requiresWrite = (lSize > _limit        //global buffer limit
			|| !ByteBuffer.isValidCapacity(lSize, cb)); //local buffer limit
		int numEvicted = 0;

		//handle caching/eviction if it fits in writebuffer
		if( !requiresWrite )
		{
			//create byte buffer handle (no block allocation yet)
			ByteBuffer bbuff = new ByteBuffer( lSize );

			//modify buffer pool
			synchronized( _mQueue )
			{
				//evict matrices to make room (by default FIFO)
				while( _size+lSize > _limit && !_mQueue.isEmpty() )
				{
					//remove first entry from eviction queue
					Map.Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
					String ftmp = entry.getKey();
					ByteBuffer tmp = entry.getValue();

					if( tmp != null ) {
						//wait for pending serialization
						tmp.checkSerialized();

						//evict matrix
						tmp.evictBuffer(ftmp);
						tmp.freeMemory();
						_size -= tmp.getSize();
						numEvicted++;
					}
				}

				//put placeholder into buffer pool (reserve mem)
				_mQueue.addLast(fname, bbuff);
				_size += lSize;
			}

			//serialize matrix (outside synchronized critical path)
			_fClean.serializeData(bbuff, cb);

			if( DMLScript.STATISTICS ) {
				CacheStatistics.incrementFSBuffWrites();
				CacheStatistics.incrementFSWrites(numEvicted);
			}
		}
		else
		{
			//write directly to local FS (bypass buffer if too large)
			LocalFileUtils.writeCacheBlockToLocal(fname, cb);
			if( DMLScript.STATISTICS ) {
				CacheStatistics.incrementFSWrites();
			}
			numEvicted++;
		}

		return numEvicted;
	}

	public static long getCacheBlockSize(CacheBlock cb) {
		return cb.isShallowSerialize() ?
			cb.getInMemorySize() : cb.getExactSerializedSize();
	}

	public static void deleteBlock(String fname)
	{
		boolean requiresDelete = true;

		synchronized( _mQueue )
		{
			//remove queue entry
			ByteBuffer ldata = _mQueue.remove(fname);
			if( ldata != null ) {
				_size -= ldata.getSize();
				requiresDelete = false;
				ldata.freeMemory(); //cleanup
			}
		}

		//delete from FS if required
		if( requiresDelete )
			_fClean.deleteFile(fname);
	}

	/**
	 * Evicts all buffer pool entries.
	 * NOTE: use only for debugging or testing.
	 *
	 * @throws IOException if IOException occurs
	 */
	public static void forceEviction()
		throws IOException
	{
		//evict all matrices and frames
		while( !_mQueue.isEmpty() )
		{
			//remove first entry from eviction queue
			Map.Entry<String, ByteBuffer> entry = _mQueue.removeFirst();
			ByteBuffer tmp = entry.getValue();

			if( tmp != null ) {
				//wait for pending serialization
				tmp.checkSerialized();

				//evict matrix
				tmp.evictBuffer(entry.getKey());
				tmp.freeMemory();
			}
		}
	}
}
