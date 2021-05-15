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

import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class LineageGPUCacheEviction 
{
	private static long _currentCacheSize = 0;
	private static long GPU_CACHE_LIMIT; //limit in bytes
	private static long _startTimestamp = 0;
	public static ExecutorService gpuEvictionThread = null;
	private static TreeSet<LineageCacheEntry> weightedQueue = new TreeSet<>(LineageCacheConfig.LineageCacheComparator);

	protected static void resetEviction() {
		while(!weightedQueue.isEmpty()) {
			LineageCacheEntry e = weightedQueue.pollFirst();
			e._gpuObject.setIsLinCached(false);
			e._gpuObject.clearData(null, true);
		}
		_currentCacheSize = 0;
		gpuEvictionThread = null;
		//LineageCacheConfig.CONCURRENTGPUEVICTION = false;
		weightedQueue.clear();
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
	
	private static void adjustD2HTransferSpeed(double sizeByte, double copyTime) {
		double sizeMB = sizeByte / (1024*1024);
		double newTSpeed = sizeMB / copyTime;  //bandwidth (MB/sec) + java overhead

		// FIXME: A D2H copy lazily executes previous kernels
		if (newTSpeed > LineageCacheConfig.D2HMAXBANDWIDTH)
			return;  //filter out errorneous measurements (~ >8GB/sec)
		// Perform exponential smoothing.
		double smFactor = 0.5;  //smoothing factor
		LineageCacheConfig.D2HCOPY = (smFactor * newTSpeed) + ((1-smFactor) * LineageCacheConfig.D2HCOPY);
		//System.out.println("size_t: "+sizeMB+ " speed_t: "+newTSpeed + " estimate_t+1: "+LineageCacheConfig.D2HCOPY);
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
		weightedQueue.add(entry);
	}
	
	public static boolean isGPUCacheEmpty() {
		return weightedQueue.isEmpty();
	}

	public static LineageCacheEntry pollFirstEntry() {
		return weightedQueue.pollFirst();
	}

	public static LineageCacheEntry peekFirstEntry() {
		return weightedQueue.first();
	}
	
	public static void removeEntry(LineageCacheEntry e) {
		weightedQueue.remove(e);
	}

	public static void addEntryList(List<LineageCacheEntry> entryList) {
		weightedQueue.addAll(entryList);
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
	
	public static void copyToHostCache(LineageCacheEntry entry, String instName, boolean alreadyCopied) {
		// TODO: move to the shadow buffer. Convert to double precision only when reused.
		long t0 = System.nanoTime();
		MatrixBlock mb = alreadyCopied ? entry._gpuObject.getMatrixObject().acquireReadAndRelease()
				: entry._gpuObject.evictFromDeviceToHostMB(instName, false);
		long t1 = System.nanoTime();
		adjustD2HTransferSpeed(((double)entry._gpuObject.getSizeOnDevice()), ((double)(t1-t0))/1000000000);
		long size = mb.getInMemorySize();
		// make space in the host memory for the data TODO: synchronize
		if (!LineageCacheEviction.isBelowThreshold(size)) {
			synchronized (LineageCache.getLineageCache()) {
				LineageCacheEviction.makeSpace(LineageCache.getLineageCache(), size);
			}
		}
		// FIXME: updateSize outside of synchronized is problematic, but eliminates waiting for background eviction
		LineageCacheEviction.updateSize(size, true);
		// place the data and set gpu object to null in the cache entry
		entry.setValue(mb);
		// maintain order for eviction of host cache. FIXME: synchronize
		LineageCacheEviction.addEntry(entry);
		// manage space in gpu cache
		updateSize(size, false);
	}

}