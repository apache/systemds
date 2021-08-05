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

package org.apache.sysds.runtime.instructions.gpu.context;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheEntry;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.lineage.LineageGPUCacheEviction;
import org.apache.sysds.utils.GPUStatistics;

public class GPUMemoryEviction implements Runnable 
{
	int numEvicts;
	
	public GPUMemoryEviction(int num) {
		numEvicts = num;
	}
	
	public GPUMemoryEviction() {
		numEvicts = 0;
	}

	@Override
	public void run() {
		//long currentAvailableMemory = allocator.getAvailableMemory();
		List<LineageCacheEntry> lockedOrLiveEntries = new ArrayList<>();
		int count = 0;

		// Stop if 1) Evicted the request number of entries, 2) The parallel
		// CPU instruction is ended, and 3) No non-live entries left in the cache.
		long t0 =  DMLScript.STATISTICS ? System.nanoTime() : 0;
		while (!LineageGPUCacheEviction.isGPUCacheEmpty()) 
		{
			if (LineageCacheConfig.STOPBACKGROUNDEVICTION)
				// This logic reduces #evictions if the cpu instructions is so small
				// that it ends before the background thread reaches this condition.
				// However, this check decreases race conditions.
				break;
			
			if (numEvicts > 0 && count > numEvicts)
				break;
			
			LineageCacheEntry le = LineageGPUCacheEviction.pollFirstEntry();
			GPUObject cachedGpuObj = le.getGPUObject();
			GPUObject headGpuObj = cachedGpuObj.lineageCachedChainHead != null
					? cachedGpuObj.lineageCachedChainHead : cachedGpuObj;
			// Check and continue if any object in the linked list is locked
			boolean lockedOrLive = false;
			GPUObject nextgpuObj = headGpuObj;
			while (nextgpuObj!= null) {
				if (!nextgpuObj.isrmVarPending() || nextgpuObj.isLocked()) // live or locked
					lockedOrLive = true;
				nextgpuObj = nextgpuObj.nextLineageCachedEntry;
			}
			if (lockedOrLive) {
				lockedOrLiveEntries.add(le);
				continue;
			}

			// TODO: First remove the gobj chains that don't contain any live and dirty objects.
			//currentAvailableMemory += headGpuObj.getSizeOnDevice();

			// Copy from device to host for all live and dirty objects
			boolean copied = false;
			nextgpuObj = headGpuObj;
			while (nextgpuObj!= null) {
				// Keeping isLinCached as True here will save data deletion by copyFromDeviceToHost
				if (!nextgpuObj.isrmVarPending() && nextgpuObj.isDirty()) { //live and dirty
					nextgpuObj.copyFromDeviceToHost(null, true, true);
					copied = true;
				}
				nextgpuObj.setIsLinCached(false);
				nextgpuObj = nextgpuObj.nextLineageCachedEntry;
			}

			// Copy from device cache to CPU lineage cache if not already copied
			LineageGPUCacheEviction.copyToHostCache(le, null, copied);

			// For all the other objects, remove and clear data (only once)
			nextgpuObj = headGpuObj;
			boolean freed = false;
			synchronized (nextgpuObj.getGPUContext().getMemoryManager().getGPUMatrixMemoryManager().gpuObjects) {
				while (nextgpuObj!= null) {
					// If not live or live but not dirty
					if (nextgpuObj.isrmVarPending() || !nextgpuObj.isDirty()) {
						if (!freed) {
							nextgpuObj.clearData(null, true);
							//FIXME: adding to rmVar cache causes multiple failures due to concurrent
							//access to the rmVar cache and other data structures. VariableCP instruction
							//and other instruction free memory and add to rmVar cache in parallel to
							//the background eviction task, which needs to be synchronized.
							freed = true;
						}
						else
							nextgpuObj.clearGPUObject();
					}
					nextgpuObj = nextgpuObj.nextLineageCachedEntry;
				}
			}
			// Clear the GPUOjects chain
			GPUObject currgpuObj = headGpuObj;
			while (currgpuObj.nextLineageCachedEntry != null) {
				nextgpuObj = currgpuObj.nextLineageCachedEntry;
				currgpuObj.lineageCachedChainHead = null;
				currgpuObj.nextLineageCachedEntry = null;
				nextgpuObj.lineageCachedChainHead = null;
				currgpuObj = nextgpuObj;
			}

			//if(currentAvailableMemory >= size)
				// This doesn't guarantee allocation due to fragmented freed memory
			//	A = cudaMallocNoWarn(tmpA, size, null); 
			if (DMLScript.STATISTICS) {
				LineageCacheStatistics.incrementGpuAsyncEvicts();
			}
			count++;
		}

		// Add the locked entries back to the eviction queue
		if (!lockedOrLiveEntries.isEmpty())
			LineageGPUCacheEviction.addEntryList(lockedOrLiveEntries);
		
		if (DMLScript.STATISTICS) //TODO: dedicated statistics for lineage
			GPUStatistics.cudaEvictTime.add(System.nanoTime() - t0);
	}
}
