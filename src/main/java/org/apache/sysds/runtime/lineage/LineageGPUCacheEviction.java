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

package org.apache.sysds.runtime.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;

import jcuda.Pointer;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import static org.apache.sysds.runtime.instructions.gpu.context.GPUObject.toIntExact;

public class LineageGPUCacheEviction 
{
	private static long _currentCacheSize = 0;
	private static long GPU_CACHE_LIMIT; //limit in bytes
	private static GPUContext _gpuContext = null;
	private static long _startTimestamp = 0;
	public static ExecutorService gpuEvictionThread = null;

	// Weighted queue of freed pointers.
	private static HashMap<Long, TreeSet<LineageCacheEntry>> freeQueues = new HashMap<>();

	// Pointers and live counts associated
	private static HashMap<Pointer, Integer> livePointers = new HashMap<>();

	// All cached pointers mapped to the corresponding lineage cache entries
	private static HashMap<Pointer, LineageCacheEntry> GPUCacheEntries = new HashMap<>();

	protected static void resetEviction() {
		_currentCacheSize = 0;
		gpuEvictionThread = null;
		//LineageCacheConfig.CONCURRENTGPUEVICTION = false;
		freeQueues.clear();
		livePointers.clear();
		GPUCacheEntries.clear();
	}

	public static void setGPUContext(GPUContext gpuCtx) {
		_gpuContext = gpuCtx;
	}

	protected static GPUContext getGPUContext() {
		return _gpuContext;
	}

	protected static long getPointerSize(Pointer ptr) {
		return _gpuContext.getMemoryManager().getSizeAllocatedGPUPointer(ptr);
	}

	protected static void incrementLiveCount(Pointer ptr) {
		// Move from free list (if exists) to live list
		if(livePointers.merge(ptr, 1, Integer::sum) == 1)
			freeQueues.get(getPointerSize(ptr)).remove(GPUCacheEntries.get(ptr));
	}

	public static void decrementLiveCount(Pointer ptr) {
		// Decrement and move to the free list if the live count becomes 0
		if(livePointers.compute(ptr, (k, v) -> v==1 ? null : v-1) == null) {
			long size = getPointerSize(ptr);
			if (!freeQueues.containsKey(size))
				freeQueues.put(size, new TreeSet<>(LineageCacheConfig.LineageCacheComparator));
				//FIXME: Multiple entries can point to same pointer due to multi-level reuse
			freeQueues.get(size).add(GPUCacheEntries.get(ptr));
		}
	}

	public static boolean probeLiveCachedPointers(Pointer ptr) {
		return livePointers.containsKey(ptr);
	}

	//---------------- COSTING RELATED METHODS -----------------

	/**
	 * Set the max constraint for the lineage cache in GPU
	 */
	public static void setGPULineageCacheLimit() {
		long available = GPUContextPool.initialGPUMemBudget();
		GPU_CACHE_LIMIT = (long) (available * LineageCacheConfig.GPU_CACHE_MAX);
	}
	protected static void setStartTimestamp() {
		_startTimestamp = System.currentTimeMillis();
	}

	protected static long getStartTimestamp() {
		return _startTimestamp;
	}
	
	@SuppressWarnings("unused")
	private static void adjustD2HTransferSpeed(double sizeByte, double copyTime) {
		double sizeMB = sizeByte / (1024*1024);
		double newTSpeed = sizeMB / copyTime;  //bandwidth (MB/sec) + java overhead

		if (newTSpeed > LineageCacheConfig.D2HMAXBANDWIDTH)
			return;  //filter out errorneous measurements (~ >8GB/sec)
		// Perform exponential smoothing.
		double smFactor = 0.5;  //smoothing factor
		LineageCacheConfig.D2HCOPYBANDWIDTH = (smFactor * newTSpeed) + ((1-smFactor) * LineageCacheConfig.D2HCOPYBANDWIDTH);
		//System.out.println("size_t: "+sizeMB+ " speed_t: "+newTSpeed + " estimate_t+1: "+LineageCacheConfig.D2HCOPYBANDWIDTH);
	}

	//--------------- CACHE MAINTENANCE & LOOKUP FUNCTIONS --------------//

	protected static void addEntry(LineageCacheEntry entry) {
		if (entry.isNullVal())
			// Placeholders shouldn't participate in eviction cycles.
			return;
		if (entry.isScalarValue())
			throw new DMLRuntimeException ("Scalars are never stored in GPU. Lineage: "+ entry._key);

		// TODO: Separate removelist, starttimestamp, score and weights from CPU cache
		entry.computeScore(LineageCacheEviction._removelist);
		// The pointer must be live at this moment
		livePointers.put(entry.getGPUPointer(), 1);
		GPUCacheEntries.put(entry.getGPUPointer(), entry);
	}
	
	public static boolean isGPUCacheEmpty() {
		return (freeQueues.isEmpty() && livePointers.isEmpty());
	}

	public static boolean isGPUCacheFreeQEmpty() {
		return freeQueues.isEmpty();
	}

	// Remove and return the cached free pointer with exact size
	public static LineageCacheEntry pollFirstFreeEntry(long size) {
		TreeSet<LineageCacheEntry> freeList = freeQueues.get(size);
		if (freeList != null && freeList.isEmpty())
			freeQueues.remove(size); //remove if empty

		// Poll the first pointer from the queue
		if (freeList != null && !freeList.isEmpty())
			return freeList.pollFirst();
		return null;
	}

	// Remove and return the minimum non-exact sized pointer.
	// If no bigger sized pointer available, return one from the highest sized list
	public static LineageCacheEntry pollFistFreeNotExact(long size) {
		// Assuming no exact match
		List<Long> sortedSizes = new ArrayList<>(freeQueues.keySet());
		// If the asked size is bigger than all, return a pointer of the highest size available
		long maxSize = sortedSizes.get(sortedSizes.size()-1);
		if (size > maxSize)
			return pollFirstFreeEntry(maxSize);
		// Return a pointer of the next biggest size
		for (long fSize : sortedSizes) {
			if (fSize >= size)
				return pollFirstFreeEntry(fSize);
		}
		return null;
	}

	public static LineageCacheEntry peekFirstFreeEntry(long size) {
		return freeQueues.get(size).first();
	}
	
	public static void removeFreeEntry(LineageCacheEntry e) {
		long size = getPointerSize(e.getGPUPointer());
		freeQueues.get(size).remove(e);
	}

	//---------------- CACHE SPACE MANAGEMENT METHODS -----------------//

	protected static void updateSize(long space, boolean addspace) {
		if (addspace)
			_currentCacheSize += space;
		else
			_currentCacheSize -= space;
	}

	protected static boolean isBelowMaxThreshold(long spaceNeeded) {
		return ((spaceNeeded + _currentCacheSize) <= GPU_CACHE_LIMIT);
	}
	
	protected static long getGPUCacheLimit() {
		return GPU_CACHE_LIMIT;
	}

	public static int numPointersCached() {
		return livePointers.size() + freeQueues.values().stream().mapToInt(TreeSet::size).sum();
	}

	public static long totalMemoryCached() {
		long totLive = livePointers.keySet().stream()
			.mapToLong(ptr -> _gpuContext.getMemoryManager().getSizeAllocatedGPUPointer(ptr)).sum();
		long totFree = 0;
		for (Map.Entry<Long, TreeSet<LineageCacheEntry>> entry : freeQueues.entrySet())
			totFree += entry.getKey() * entry.getValue().size();
		return totLive + totFree;
	}

	public static Set<Pointer> getAllCachedPointers() {
		Set<Pointer> cachedPointers = new HashSet<>();
		for (Map.Entry<Long, TreeSet<LineageCacheEntry>> entry : freeQueues.entrySet())
			cachedPointers.addAll(entry.getValue().stream()
				.map(LineageCacheEntry::getGPUPointer).collect(Collectors.toSet()));
		cachedPointers.addAll(livePointers.keySet());
		return cachedPointers;
	}

	// Copy an intermediate from GPU cache to host cache
	// TODO: move to the shadow buffer. Convert to double precision only when reused.
	public static Pointer copyToHostCache(LineageCacheEntry entry) {
		// Memcopy from the GPU pointer to a matrix block
		long t0 = System.nanoTime();
		MatrixBlock mb = pointerToMatrixBlock(entry);
		long t1 = System.nanoTime();
		// Adjust the estimated D2H bandwidth
		adjustD2HTransferSpeed(((double)entry.getSize()), ((double)(t1-t0))/1000000000);
		Pointer ptr = entry.getGPUPointer();
		long size = mb.getInMemorySize();
		synchronized(LineageCache.getLineageCache()) {
			// Make space in the host cache for the data
			if(!LineageCacheEviction.isBelowThreshold(size)) {
				synchronized(LineageCache.getLineageCache()) {
					LineageCacheEviction.makeSpace(LineageCache.getLineageCache(), size);
				}
			}
			LineageCacheEviction.updateSize(size, true);
			// Place the data and set gpu object to null in the cache entry
			entry.setValue(mb);
			// Maintain order for eviction of host cache.
			LineageCacheEviction.addEntry(entry);
		}
		return ptr;
	}

	private static MatrixBlock pointerToMatrixBlock(LineageCacheEntry le) {
		MatrixBlock ret = null;
		DataCharacteristics dc = le.getDataCharacteristics();
		if (le.isDensePointer()) {
			ret = new MatrixBlock(toIntExact(dc.getRows()), toIntExact(dc.getCols()), false);
			ret.allocateDenseBlock();
			// copy to the host
			LibMatrixCUDA.cudaSupportFunctions.deviceToHost(getGPUContext(),
				le.getGPUPointer(), ret.getDenseBlockValues(), null, true);
			ret.recomputeNonZeros();
		} /*else {
			int rows = toIntExact(dc.getRows());
			int cols = toIntExact(dc.getCols());
			int nnz = toIntExact(le.getGPUPointer().nnz);
			double[] values = new double[nnz];
			LibMatrixCUDA.cudaSupportFunctions.deviceToHost(getGPUContext(), le.getGPUPointer().val, values, null, true);
			int[] rowPtr = new int[rows + 1];
			int[] colInd = new int[nnz];
			CSRPointer.copyPtrToHost(le.getGPUPointer(), rows, nnz, rowPtr, colInd);
			SparseBlockCSR sparseBlock = new SparseBlockCSR(rowPtr, colInd, values, nnz);
			ret = new MatrixBlock(rows, cols, nnz, sparseBlock);
		}*/
		//mat.acquireModify(tmp);
		//mat.release();
		return ret;
	}

	public static void removeFromDeviceCache(LineageCacheEntry entry, Pointer ptr, boolean removeFromCache) {
		long size = _gpuContext.getMemoryManager().getSizeAllocatedGPUPointer(ptr);
		if (removeFromCache)
			LineageCache.removeEntry(entry._key);
		updateSize(size, false);
		GPUCacheEntries.remove(ptr);
	}

}