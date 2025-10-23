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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.prescientbuffer.PrescientPolicy;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
 * configured internally. A good starting point are the following constraints (relative to
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
 *  	and output. Evict cached objects to make space for the inputs.
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
	// Maximum size of UMM in bytes (default 85%)
	private static long _limit;
	// Current total size of the cached objects
	private static long _totCachedSize;
	// Operational memory limit in bytes (70%)
	private static long _opMemLimit;
	// List of pinned entries
	private static final List<String> _pinnedEntries = new ArrayList<>();

	// Eviction queue of <filename,buffer> pairs (implemented via linked hash map
	// for (1) queue semantics and (2) constant time get/insert/delete operations)
	private static CacheEvictionQueue _mQueue;

	// Maintenance service for synchronous or asynchronous delete of evicted files
	private static CacheMaintenanceService _fClean;

	// Prescient policy
	private static PrescientPolicy _prescientPolicy;
//	private static IOTrace _ioTrace;

	// Pinned size of physical memory. Starts from 0 for each operation. Max is 70% of heap
	// This increases only if the input is not present in the cache and read from FS/rdd/fed/gpu
	private static long _pinnedPhysicalMemSize = 0;
	// Size of pinned virtual memory. This tracks the total input size
	// This increases if the input is available in the cache.
	private static long _pinnedVirtualMemSize = 0;

	//---------------- OPERATION MEMORY MAINTENANCE -------------------//

	// Make space for and track a cache block to be pinned in operation memory
	public static void pin(CacheableData<?> cd) {
		if (!CacheableData.isCachingActive()) {
			return;
		}

		// Space accounting based on an estimated size and before reading the blob
		long estimatedSize = OptimizerUtils.estimateSize(cd.getDataCharacteristics());
		if (probe(cd))
			// Availability in the cache means no memory overhead.
			// We still need to track to derive the worst-case output memory
			_pinnedVirtualMemSize += estimatedSize;
		else {
			// The blob will be restored from local FS, or will be read
			// from other backends. Make space if not available.
			makeSpace(estimatedSize);
			_pinnedPhysicalMemSize += estimatedSize;
		}
		// Track the pinned entries to protect from evictions
		_pinnedEntries.add(cd.getCacheFilePathAndName());

		// Reserve space for output after pinning every input.
		// This overly conservative approach removes the need to call reserveOutputMem() from
		// each instruction. Ideally, every instruction first pins all the inputs, followed
		// by reserving space for the output.
		reserveOutputMem();
	}

	// Reserve space for output in the operation memory
	public static void reserveOutputMem() {
		if (!OptimizerUtils.isUMMEnabled() || !CacheableData.isCachingActive())
			return;

		// Worst case upper bound for output = 70% - size(inputs)
		// FIXME: Parfor splits this 70% into smaller limits
		long maxOutputSize = _opMemLimit - (_pinnedVirtualMemSize + _pinnedPhysicalMemSize);
		// Evict cached entries to make space in operation memory if needed
		makeSpace(maxOutputSize);
	}
	
	// Unpins (releases) a cache block from operation memory
	public static void unpin(CacheableData<?> cd) {
		if (!CacheableData.isCachingActive())
			return;

		// TODO: Track preserved output memory to protect from concurrent threads
		if (!_pinnedEntries.contains(cd.getCacheFilePathAndName()))
			return; //unpinned. output of an instruction

		// We still use the estimated size even though we have the blobs available.
		// This makes sure we are subtracting exactly what we added during pinning.
		long estimatedSize = OptimizerUtils.estimateSize(cd.getDataCharacteristics());
		if (probe(cd))
			_pinnedVirtualMemSize -= estimatedSize;
		else
			_pinnedPhysicalMemSize -= estimatedSize;

		_pinnedEntries.remove(cd.getCacheFilePathAndName());
	}

	//---------------- UMM MAINTENANCE & LOOKUP -------------------//

	// Initialize the unified memory manager
	public static void init() {
		_mQueue = new CacheEvictionQueue();
		_fClean = new CacheMaintenanceService();
		_limit = OptimizerUtils.getBufferPoolLimit();
		_opMemLimit = (long)(OptimizerUtils.getLocalMemBudget()); //70% of heap
		_totCachedSize = 0;
		_pinnedPhysicalMemSize = 0;
		_pinnedVirtualMemSize = 0;
	}

	// Cleanup the unified memory manager
	public static void cleanup() {
		if( _mQueue != null )
			_mQueue.clear();
		if( _fClean != null )
			_fClean.close();
		_totCachedSize = 0;
		_pinnedPhysicalMemSize = 0;
		_pinnedVirtualMemSize = 0;
	}

	/**
	 * Print current status of UMM, including all entries.
	 * NOTE: use only for debugging or testing.
	 *
	 * @param operation e.g. BEFORE PIN, AFTER UNPIN, AT MAKESPACE
	 */
	public static void printStatus(String operation)
	{
		System.out.println("UMM STATUS AT "+operation+" --");

		synchronized (_mQueue) {
			// print UMM meta data
			System.out.println("\tUMM: Meta Data: " +
				"UMM limit="+_limit+", " +
				"size[bytes]="+_totCachedSize+", " +
				"size[elements]="+_mQueue.size()+", " +
				"pinned[elements]="+_pinnedEntries.size()+", " +
				"pinned[bytes]="+_pinnedPhysicalMemSize);

			// print current cached entries
			int count = _mQueue.size();
			for (Map.Entry<String, ByteBuffer> entry : _mQueue.entrySet()) {
				String fname = entry.getKey();
				ByteBuffer bbuff = entry.getValue();
				System.out.println("\tUMM: Cached element ("+count+"): "
					+fname+", "+(bbuff.isShallow()?bbuff._cdata.getClass().getSimpleName():"?")
					+", "+bbuff.getSize()+", "+bbuff.isShallow());
				count--;
			}
		}
	}

	public static void setUMMLimit(long val) {
		_limit = val;
	}

	public static long getUMMSize() {
		synchronized(_mQueue) {
			return _limit;
		}
	}

	// Get the available memory in UMM
	public static long getUMMFree() {
		synchronized(_mQueue) {
			return _limit - (_totCachedSize + _pinnedPhysicalMemSize);
		}
	}

	// Reads a cached object. This is called from cacheabledata implementations
	public static CacheBlock<?> readBlock(String fname, boolean matrix)
		throws IOException
	{
		CacheBlock<?> cb = null;
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

	// Make required space. Evict if needed.
	public static int makeSpace(long reqSpace) {
		int numEvicted = 0;
		// Check if sufficient space is already available
		if (getUMMFree() > reqSpace)
			return numEvicted;

		// Evict cached objects to make space
		try {
			synchronized(_mQueue) {
				// Evict blobs to make room (by default FIFO)
				while (getUMMFree() < reqSpace && !_mQueue.isEmpty()) {
					//remove first unpinned entry from eviction queue
					var entry = _mQueue.removeFirstUnpinned(_pinnedEntries);
					String ftmp = entry.getKey();
					ByteBuffer bb = entry.getValue();

					if(bb != null) {
						// Wait for pending serialization
						bb.checkSerialized();
						// Evict object
						bb.evictBuffer(ftmp);
						bb.freeMemory();
						_totCachedSize -= bb.getSize();
						numEvicted++;
					}
				}
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Eviction request of size "+(reqSpace-getUMMFree())+ " in the UMM failed.", e);
		}

		if( DMLScript.STATISTICS )
			CacheStatistics.incrementFSWrites(numEvicted);

		return numEvicted;
	}

	// Write an object to the cache
	public static int writeBlock(String fname, CacheBlock<?> cb)
		throws IOException
	{
		//obtain basic metadata of the cache block
		long lSize = getCacheBlockSize(cb);
		boolean requiresWrite = (lSize > _limit        //global buffer limit
			|| !ByteBuffer.isValidCapacity(lSize, cb)); //local buffer limit
		int numEvicted = 0;

		// Handle caching/eviction if it fits in UMM
		if( !requiresWrite )
		{
			// Create byte buffer handle (no block allocation yet)
			ByteBuffer bbuff = new ByteBuffer( lSize );

			// Modify buffer pool
			synchronized( _mQueue )
			{
				// Evict blocks to make room if required
				numEvicted += makeSpace(lSize);
				// Put placeholder into buffer pool (reserve mem)
				_mQueue.addLast(fname, bbuff);
				_totCachedSize += lSize;
			}

			// Serialize matrix (outside synchronized critical path)
			_fClean.serializeData(bbuff, cb);

			if( DMLScript.STATISTICS )
				CacheStatistics.incrementBPoolWrites();
		}
		else
		{
			// Write directly to local FS (bypass buffer if too large)
			LocalFileUtils.writeCacheBlockToLocal(fname, cb);
			if( DMLScript.STATISTICS ) {
				CacheStatistics.incrementFSWrites();
			}
			numEvicted++;
		}

		return numEvicted;
	}

	public static long getCacheBlockSize(CacheBlock<?> cb) {
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
				_totCachedSize -= ldata.getSize();
				requiresDelete = false;
				ldata.freeMemory(); //cleanup
			}
		}

		//delete from FS if required
		if( requiresDelete )
			_fClean.deleteFile(fname);
	}

	/**
	 * Removes all cache blocks from all memory areas and deletes all evicted
	 * representations (files in local FS). All internally thread pools must be
	 * shut down in a graceful manner (e.g., wait for pending deletes).
	 */
	public void deleteAll() {
		//TODO implement
		throw new NotImplementedException();
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
