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
package org.tugraz.sysds.runtime.instructions.gpu.context;

import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.JCuda.cudaMemset;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.instructions.gpu.GPUInstruction;
import org.tugraz.sysds.utils.GPUStatistics;

import jcuda.Pointer;
/**
 * - All cudaFree and cudaMalloc in SystemDS should go through this class to avoid OOM or incorrect results.
 * - This class can be refactored in future to accept a chunk of memory ahead of time rather than while execution. This will only thow memory-related errors during startup.  
 */
public class GPUMemoryManager {
	protected static final Log LOG = LogFactory.getLog(GPUMemoryManager.class.getName());
	
	// Developer flag: Use this flag to check for GPU memory leak in SystemDS.
	// This has an additional overhead of maintaining stack trace of all the allocated GPU pointers via PointerInfo class.
	private static final boolean DEBUG_MEMORY_LEAK = false;
	private static final int [] DEBUG_MEMORY_LEAK_STACKTRACE_DEPTH = {5, 6, 7, 8, 9, 10}; // Avoids printing too much text while debuggin
	
	protected final GPUMemoryAllocator allocator;
	/*****************************************************************************************/
	// GPU Memory is divided into three major sections:
	// 1. Matrix Memory: Memory allocated to matrices in SystemDS and addressable by GPUObjects.
	// This memory section is divided into three minor sections:
	// 1.1 Locked Matrix Memory
	// 1.2 UnLocked + Non-Dirty Matrix Memory
	// 1.3 UnLocked + Dirty Matrix Memory
	// To get the GPUObjects/Pointers in this section, please use getGPUObjects and getPointers methods of GPUMatrixMemoryManager.
	// To clear GPUObjects/Pointers in this section, please use clear and clearAll methods of GPUMatrixMemoryManager.
	// Both these methods allow to get/clear unlocked/locked and dirty/non-dirty objects of a certain size.
	protected final GPUMatrixMemoryManager matrixMemoryManager;
	public GPUMatrixMemoryManager getGPUMatrixMemoryManager() {
		return matrixMemoryManager;
	}
	
	// 2. Rmvar-ed pointers: If sysds.gpu.eager.cudaFree is set to false,
	// then this manager caches pointers of the GPUObject on which rmvar instruction has been executed for future reuse.
	// We observe 2-3x improvement with this approach and hence recommend to set this flag to false.
	protected final GPULazyCudaFreeMemoryManager lazyCudaFreeMemoryManager;
	public GPULazyCudaFreeMemoryManager getGPULazyCudaFreeMemoryManager() {
		return lazyCudaFreeMemoryManager;
	}
	
	// 3. Non-matrix locked pointers: Other pointers (required for execution of an instruction that are not memory). For example: workspace
	// These pointers are not explicitly tracked by a memory manager but one can get them by using getNonMatrixLockedPointers
	private Set<Pointer> getNonMatrixLockedPointers() {
		Set<Pointer> managedPointers = matrixMemoryManager.getPointers();
		managedPointers.addAll(lazyCudaFreeMemoryManager.getAllPointers());
		return nonIn(allPointers.keySet(), managedPointers);
	}
	
	
	/**
	 * To record size of all allocated pointers allocated by above memory managers
	 */
	protected final HashMap<Pointer, PointerInfo> allPointers = new HashMap<>();
	
	/*****************************************************************************************/
	

	/**
	 * Get size of allocated GPU Pointer
	 * @param ptr pointer to get size of
	 * @return either the size or -1 if no such pointer exists
	 */
	public long getSizeAllocatedGPUPointer(Pointer ptr) {
		if(allPointers.containsKey(ptr)) {
			return allPointers.get(ptr).getSizeInBytes();
		}
		return -1;
	}
	
	/**
	 * Utility to debug memory leaks
	 */
	static class PointerInfo {
		private long sizeInBytes;
		private StackTraceElement[] stackTraceElements;
		public PointerInfo(long sizeInBytes) {
			if(DEBUG_MEMORY_LEAK) {
				this.stackTraceElements = Thread.currentThread().getStackTrace();
			}
			this.sizeInBytes = sizeInBytes;
		}
		public long getSizeInBytes() {
			return sizeInBytes;
		}
	}
	
	// If the available free size is less than this factor, GPUMemoryManager will warn users of multiple programs grabbing onto GPU memory.
	// This often happens if user tries to use both TF and SystemDS, and TF grabs onto 90% of the memory ahead of time.
	private static final double WARN_UTILIZATION_FACTOR = 0.7;
	
	public GPUMemoryManager(GPUContext gpuCtx) {
		matrixMemoryManager = new GPUMatrixMemoryManager(this);
		lazyCudaFreeMemoryManager = new GPULazyCudaFreeMemoryManager(this);
		if(DMLScript.GPU_MEMORY_ALLOCATOR.equals("cuda")) {
			allocator = new CudaMemoryAllocator();
		}
		else if(DMLScript.GPU_MEMORY_ALLOCATOR.equals("unified_memory")) {
			allocator = new UnifiedMemoryAllocator();
		}
		else {
			throw new RuntimeException("Unsupported value (" + DMLScript.GPU_MEMORY_ALLOCATOR + ") for the configuration " + DMLConfig.GPU_MEMORY_ALLOCATOR 
					+ ". Supported values are cuda, unified_memory.");
		}
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		if(free[0] < WARN_UTILIZATION_FACTOR*total[0]) {
			LOG.warn("Potential under-utilization: GPU memory - Total: " + (total[0] * (1e-6)) + " MB, Available: " + (free[0] * (1e-6)) + " MB on " + gpuCtx 
					+ ". This can happen if there are other processes running on the GPU at the same time.");
		}
		else {
			LOG.info("GPU memory - Total: " + (total[0] * (1e-6)) + " MB, Available: " + (free[0] * (1e-6)) + " MB on " + gpuCtx);
		}
		if (GPUContextPool.initialGPUMemBudget() > OptimizerUtils.getLocalMemBudget()) {
			LOG.warn("Potential under-utilization: GPU memory (" + GPUContextPool.initialGPUMemBudget()
					+ ") > driver memory budget (" + OptimizerUtils.getLocalMemBudget() + "). "
					+ "Consider increasing the driver memory budget.");
		}
	}
	
	/**
	 * Invoke cudaMalloc
	 * 
	 * @param A pointer
	 * @param size size in bytes
	 * @param printDebugMessage debug message
	 * @return allocated pointer
	 */
	private Pointer cudaMallocNoWarn(Pointer A, long size, String printDebugMessage) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		try {
			allocator.allocate(A, size);
			allPointers.put(A, new PointerInfo(size));
			if(DMLScript.STATISTICS) {
				long totalTime = System.nanoTime() - t0;
				GPUStatistics.cudaAllocSuccessTime.add(totalTime);
				GPUStatistics.cudaAllocSuccessCount.increment();
				GPUStatistics.cudaAllocTime.add(totalTime);
				GPUStatistics.cudaAllocCount.increment();
			}
			if(printDebugMessage != null && (DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) )  {
				LOG.info("Success: " + printDebugMessage + ":" + byteCountToDisplaySize(size));
			}
			return A;
		} catch(jcuda.CudaException e) {
			if(DMLScript.STATISTICS) {
				long totalTime = System.nanoTime() - t0;
				GPUStatistics.cudaAllocFailedTime.add(System.nanoTime() - t0);
				GPUStatistics.cudaAllocFailedCount.increment();
				GPUStatistics.cudaAllocTime.add(totalTime);
				GPUStatistics.cudaAllocCount.increment();
			}
			if(printDebugMessage != null && (DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) )  {
				LOG.info("Failed: " + printDebugMessage + ":" + byteCountToDisplaySize(size));
				LOG.info("GPU Memory info " + printDebugMessage + ":" + toString());
			}
			return null;
		}
	}
	
	/**
	 * Pretty printing utility to debug OOM error
	 * 
	 * @param stackTrace stack trace
	 * @param index call depth
	 * @return pretty printed string
	 */
	private String getCallerInfo(StackTraceElement [] stackTrace, int index) {
		if(stackTrace.length <= index)
			return "->";
		else
			return "->" + stackTrace[index].getClassName() + "." + stackTrace[index].getMethodName() + "(" + stackTrace[index].getFileName() + ":" + stackTrace[index].getLineNumber() + ")";
	}
	
	/**
	 * Pretty printing utility to print bytes
	 * 
	 * @param numBytes number of bytes
	 * @return a human-readable display value
	 */
	private String byteCountToDisplaySize(long numBytes) {
		// return org.apache.commons.io.FileUtils.byteCountToDisplaySize(bytes); // performs rounding
	    if (numBytes < 1024) { 
	    	return numBytes + " bytes";
	    }
	    else {
		    int exp = (int) (Math.log(numBytes) / 6.931471805599453);
		    return String.format("%.3f %sB", ((double)numBytes) / Math.pow(1024, exp), "KMGTP".charAt(exp-1));
	    }
	}
	
	
	/**
	 * Allocate pointer of the given size in bytes.
	 * 
	 * @param opcode instruction name
	 * @param size size in bytes
	 * @return allocated pointer
	 */
	public Pointer malloc(String opcode, long size) {
		if(size < 0) {
			throw new DMLRuntimeException("Cannot allocate memory of size " + byteCountToDisplaySize(size));
		}
		if(DEBUG_MEMORY_LEAK) {
			LOG.info("GPU Memory info during malloc:" + toString());
		}
		
		// Step 1: First try reusing exact match in rmvarGPUPointers to avoid holes in the GPU memory
		Pointer A = lazyCudaFreeMemoryManager.getRmvarPointer(opcode, size);
		
		Pointer tmpA = (A == null) ? new Pointer() : null;
		// Step 2: Allocate a new pointer in the GPU memory (since memory is available)
		// Step 3 has potential to create holes as well as limit future reuse, hence perform this step before step 3.
		if(A == null && allocator.canAllocate(size)) {
			// This can fail in case of fragmented memory, so don't issue any warning
			A = cudaMallocNoWarn(tmpA, size, "allocate a new pointer");
		}
		
		// Step 3: Try reusing non-exact match entry of rmvarGPUPointers
		if(A == null) { 
			A = lazyCudaFreeMemoryManager.getRmvarPointerMinSize(opcode, size);
			if(A != null) {
				guardedCudaFree(A);
				A = cudaMallocNoWarn(tmpA, size, "reuse non-exact match of rmvarGPUPointers"); 
				if(A == null)
					LOG.warn("cudaMalloc failed after clearing one of rmvarGPUPointers.");
			}
		}
		
		// Step 4: Eagerly free-up rmvarGPUPointers and check if memory is available on GPU
		// Evictions of matrix blocks are expensive (as they might lead them to be written to disk in case of smaller CPU budget) 
		// than doing cuda free/malloc/memset. So, rmvar-ing every blocks (step 4) is preferred to eviction (step 5).
		if(A == null) {
			lazyCudaFreeMemoryManager.clearAll();
			if(allocator.canAllocate(size)) {
				// This can fail in case of fragmented memory, so don't issue any warning
				A = cudaMallocNoWarn(tmpA, size, "allocate a new pointer after eager free");
			}
		}
		
		// Step 5: Try eviction/clearing exactly one with size restriction
		if(A == null) {
			long t0 =  DMLScript.STATISTICS ? System.nanoTime() : 0;
			Optional<GPUObject> sizeBasedUnlockedGPUObjects = matrixMemoryManager.gpuObjects.stream()
						.filter(gpuObj -> !gpuObj.isLocked() && matrixMemoryManager.getWorstCaseContiguousMemorySize(gpuObj) >= size)
						.min((o1, o2) -> worstCaseContiguousMemorySizeCompare(o1, o2));
			if(sizeBasedUnlockedGPUObjects.isPresent()) {
				evictOrClear(sizeBasedUnlockedGPUObjects.get(), opcode);
				A = cudaMallocNoWarn(tmpA, size, null);
				if(A == null)
					LOG.warn("cudaMalloc failed after clearing/evicting based on size.");
				if(DMLScript.STATISTICS) {
					long totalTime = System.nanoTime() - t0;
					GPUStatistics.cudaEvictTime.add(totalTime);
					GPUStatistics.cudaEvictSizeTime.add(totalTime);
					GPUStatistics.cudaEvictCount.increment();
					GPUStatistics.cudaEvictSizeCount.increment();
				}
			}
		}
		
		// Step 6: Try eviction/clearing one-by-one based on the given policy without size restriction
		if(A == null) {
			long t0 =  DMLScript.STATISTICS ? System.nanoTime() : 0;
			long currentAvailableMemory = allocator.getAvailableMemory();
			boolean canFit = false;
			// ---------------------------------------------------------------
			// Evict unlocked GPU objects one-by-one and try malloc
			List<GPUObject> unlockedGPUObjects = matrixMemoryManager.gpuObjects.stream()
						.filter(gpuObj -> !gpuObj.isLocked()).collect(Collectors.toList());
			Collections.sort(unlockedGPUObjects, new EvictionPolicyBasedComparator(size));
			while(A == null && unlockedGPUObjects.size() > 0) {
				GPUObject evictedGPUObject = unlockedGPUObjects.remove(unlockedGPUObjects.size()-1);
				evictOrClear(evictedGPUObject, opcode);
				if(!canFit) {
					currentAvailableMemory += evictedGPUObject.getSizeOnDevice();
					if(currentAvailableMemory >= size)
						canFit = true;
				}
				if(canFit) {
					// Checking before invoking cudaMalloc reduces the time spent in unnecessary cudaMalloc.
					// This was the bottleneck for ResNet200 experiments with batch size > 32 on P100+Intel
					A = cudaMallocNoWarn(tmpA, size, null); 
				}
				if(DMLScript.STATISTICS) 
					GPUStatistics.cudaEvictCount.increment();
			}
			if(DMLScript.STATISTICS) {
				long totalTime = System.nanoTime() - t0;
				GPUStatistics.cudaEvictTime.add(totalTime);
			}
		}
		
		
		// Step 7: Handle defragmentation
		if(A == null) {
			LOG.warn("Potential fragmentation of the GPU memory. Forcibly evicting all ...");
			LOG.info("Before clearAllUnlocked, GPU Memory info:" + toString());
			matrixMemoryManager.clearAllUnlocked(opcode);
			LOG.info("GPU Memory info after evicting all unlocked matrices:" + toString());
			A = cudaMallocNoWarn(tmpA, size, null);
		}
		
		if(A == null) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix, requested = " + byteCountToDisplaySize(size) + ". \n "
					+ toString());
		}
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		cudaMemset(A, 0, size);
		addMiscTime(opcode, GPUStatistics.cudaMemSet0Time, GPUStatistics.cudaMemSet0Count, GPUInstruction.MISC_TIMER_SET_ZERO, t0);
		return A;
	}
	
	private int worstCaseContiguousMemorySizeCompare(GPUObject o1, GPUObject o2) {
		long ret = matrixMemoryManager.getWorstCaseContiguousMemorySize(o1) - matrixMemoryManager.getWorstCaseContiguousMemorySize(o2);
		return ret < 0 ? -1 : (ret == 0 ? 0 : 1);
	}
	
	private void evictOrClear(GPUObject gpuObj, String opcode) {
		boolean eagerDelete = true;
		if(gpuObj.isDirty()) {
			// Eviction
			gpuObj.copyFromDeviceToHost(opcode, true, eagerDelete);
		}
		else {
			// Clear without copying
			gpuObj.clearData(opcode, eagerDelete);
		}
	}
	
	// --------------- Developer Utilities to debug potential memory leaks ------------------------
	private void printPointers(Set<Pointer> pointers, StringBuilder sb) {
		HashMap<String, Integer> frequency = new HashMap<>();
		for(Pointer ptr : pointers) {
			PointerInfo ptrInfo = allPointers.get(ptr);
			String key = "";
			for(int index : DEBUG_MEMORY_LEAK_STACKTRACE_DEPTH) {
				key += getCallerInfo(ptrInfo.stackTraceElements, index);
			}
			if(frequency.containsKey(key)) {
				frequency.put(key, frequency.get(key)+1);
			}
			else {
				frequency.put(key, 1);
			}
		}
		for(Entry<String, Integer> kv : frequency.entrySet()) {
			sb.append(">>" + kv.getKey() + " => " + kv.getValue() + "\n");
		}
	}
	// --------------------------------------------------------------------------------------------

	/**
	 * Note: This method should not be called from an iterator as it removes entries from allocatedGPUPointers and rmvarGPUPointers
	 * 
	 * @param toFree pointer to call cudaFree method on
	 */
	void guardedCudaFree(Pointer toFree) {
		if(allPointers.containsKey(toFree)) {
			long size = allPointers.get(toFree).getSizeInBytes();
			if(LOG.isTraceEnabled()) {
				LOG.trace("Free-ing up the pointer of size " +  byteCountToDisplaySize(size));
			}
			allPointers.remove(toFree);
			lazyCudaFreeMemoryManager.removeIfPresent(size, toFree);
			allocator.free(toFree);
			if(DMLScript.SYNCHRONIZE_GPU)
				jcuda.runtime.JCuda.cudaDeviceSynchronize(); // Force a device synchronize after free-ing the pointer for debugging
		}
		else {
			throw new RuntimeException("Attempting to free an unaccounted pointer:" + toFree);
		}

	}
	
	/**
	 * Deallocate the pointer
	 * 
	 * @param opcode instruction name
	 * @param toFree pointer to free
	 * @param eager whether to deallocate eagerly
	 * @throws DMLRuntimeException if error occurs
	 */
	public void free(String opcode, Pointer toFree, boolean eager) throws DMLRuntimeException {
		if(LOG.isTraceEnabled())
			LOG.trace("Free-ing the pointer with eager=" + eager);
		if (eager) {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			guardedCudaFree(toFree);
			addMiscTime(opcode, GPUStatistics.cudaDeAllocTime, GPUStatistics.cudaDeAllocCount, GPUInstruction.MISC_TIMER_CUDA_FREE, t0);
		}
		else {
			if (!allPointers.containsKey(toFree)) {
				LOG.info("GPU memory info before failure:" + toString());
				throw new RuntimeException("ERROR : Internal state corrupted, cache block size map is not aware of a block it trying to free up");
			}
			long size = allPointers.get(toFree).getSizeInBytes();
			lazyCudaFreeMemoryManager.add(size, toFree);
		}
	}
	
	/**
	 * Removes the GPU object from the memory manager
	 * 
	 * @param gpuObj the handle to the GPU object
	 */
	public void removeGPUObject(GPUObject gpuObj) {
		if(LOG.isDebugEnabled())
			LOG.debug("Removing the GPU object: " + gpuObj);
		matrixMemoryManager.gpuObjects.remove(gpuObj);
	}

	
	/**
	 * Clear the allocated GPU objects
	 */
	public void clearMemory() {
		// First deallocate all the GPU objects
		for(GPUObject gpuObj : matrixMemoryManager.gpuObjects) {
			if(gpuObj.isDirty()) {
				if(LOG.isDebugEnabled())
					LOG.debug("Attempted to free GPU Memory when a block[" + gpuObj + "] is still on GPU memory, copying it back to host.");
				gpuObj.copyFromDeviceToHost(null, true, true);
			}
			else
				gpuObj.clearData(null, true);
		}
		matrixMemoryManager.gpuObjects.clear();
		
		// Then clean up remaining allocated GPU pointers 
		Set<Pointer> remainingPtr = new HashSet<>(allPointers.keySet());
		for(Pointer toFree : remainingPtr) {
			guardedCudaFree(toFree); // cleans up allocatedGPUPointers and rmvarGPUPointers as well
		}
		allPointers.clear();
	}
		
	/**
	 * Performs a non-in operation
	 * 
	 * @param superset superset of pointer
	 * @param subset subset of pointer
	 * @return pointers such that: superset - subset
	 */
	private Set<Pointer> nonIn(Set<Pointer> superset, Set<Pointer> subset) {
		Set<Pointer> ret = new HashSet<Pointer>();
		for(Pointer superPtr : superset) {
			if(!subset.contains(superPtr)) {
				ret.add(superPtr);
			}
		}
		return ret;
	}
	
	/**
	 * Clears up the memory used by non-dirty pointers.
	 */
	public void clearTemporaryMemory() {
		// To record the cuda block sizes needed by allocatedGPUObjects, others are cleared up.
		Set<Pointer> unlockedDirtyPointers = matrixMemoryManager.getPointers(false, true);
		Set<Pointer> temporaryPointers = nonIn(allPointers.keySet(), unlockedDirtyPointers);
		for(Pointer tmpPtr : temporaryPointers) {
			guardedCudaFree(tmpPtr);
		}
	}
	
	/**
	 * Convenient method to add misc timers
	 * 
	 * @param opcode opcode
	 * @param globalGPUTimer member of GPUStatistics
	 * @param globalGPUCounter member of GPUStatistics
	 * @param instructionLevelTimer member of GPUInstruction
	 * @param startTime start time
	 */
	private void addMiscTime(String opcode, LongAdder globalGPUTimer, LongAdder globalGPUCounter, String instructionLevelTimer, long startTime) {
		if(DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - startTime;
			globalGPUTimer.add(totalTime);
			globalGPUCounter.add(1);
		}
	}
	
	/**
	 * Print debugging information
	 */
	@SuppressWarnings("unused")
	public String toString() {
		long sizeOfLockedGPUObjects = 0; int numLockedGPUObjects = 0; int numLockedPointers = 0;
		long sizeOfUnlockedDirtyGPUObjects = 0; int numUnlockedDirtyGPUObjects = 0; int numUnlockedDirtyPointers = 0;
		long sizeOfUnlockedNonDirtyGPUObjects = 0; int numUnlockedNonDirtyGPUObjects = 0; int numUnlockedNonDirtyPointers = 0;
		for(GPUObject gpuObj : matrixMemoryManager.gpuObjects) {
			if(gpuObj.isLocked()) {
				numLockedGPUObjects++;
				sizeOfLockedGPUObjects += gpuObj.getSizeOnDevice();
				numLockedPointers += matrixMemoryManager.getPointers(gpuObj).size();
			}
			else {
				if(gpuObj.isDirty()) {
					numUnlockedDirtyGPUObjects++;
					sizeOfUnlockedDirtyGPUObjects += gpuObj.getSizeOnDevice();
					numUnlockedDirtyPointers += matrixMemoryManager.getPointers(gpuObj).size();
				}
				else {
					numUnlockedNonDirtyGPUObjects++;
					sizeOfUnlockedNonDirtyGPUObjects += gpuObj.getSizeOnDevice();
					numUnlockedNonDirtyPointers += matrixMemoryManager.getPointers(gpuObj).size();
				}
			}
		}
		
		
		long totalMemoryAllocated = 0;
		for(PointerInfo ptrInfo : allPointers.values()) {
			totalMemoryAllocated += ptrInfo.getSizeInBytes();
		}
		
		
		Set<Pointer> potentiallyLeakyPointers = getNonMatrixLockedPointers();
		List<Long> sizePotentiallyLeakyPointers = potentiallyLeakyPointers.stream().
				map(ptr -> allPointers.get(ptr).sizeInBytes).collect(Collectors.toList());
		long totalSizePotentiallyLeakyPointers = 0;
		for(long size : sizePotentiallyLeakyPointers) {
			totalSizePotentiallyLeakyPointers += size;
		}
		StringBuilder ret = new StringBuilder();
		if(DEBUG_MEMORY_LEAK && potentiallyLeakyPointers.size() > 0) {
			ret.append("Non-matrix pointers were allocated by:\n");
			printPointers(potentiallyLeakyPointers, ret);
		}
		ret.append("\n====================================================\n");
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "", 
				"Num Objects", "Num Pointers", "Size"));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Unlocked Dirty GPU objects", 
				numUnlockedDirtyGPUObjects, numUnlockedDirtyPointers, byteCountToDisplaySize(sizeOfUnlockedDirtyGPUObjects)));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Unlocked NonDirty GPU objects", 
				numUnlockedNonDirtyGPUObjects, numUnlockedNonDirtyPointers, byteCountToDisplaySize(sizeOfUnlockedNonDirtyGPUObjects)));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Locked GPU objects", 
				numLockedGPUObjects, numLockedPointers, byteCountToDisplaySize(sizeOfLockedGPUObjects)));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Cached rmvar-ed pointers", 
				"-", lazyCudaFreeMemoryManager.getNumPointers(), byteCountToDisplaySize(lazyCudaFreeMemoryManager.getTotalMemoryAllocated())));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Non-matrix/non-cached pointers", 
				"-", potentiallyLeakyPointers.size(), byteCountToDisplaySize(totalSizePotentiallyLeakyPointers)));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "All pointers", 
				"-", allPointers.size(), byteCountToDisplaySize(totalMemoryAllocated)));
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Free mem (from cudaMemGetInfo)", 
				"-", "-", byteCountToDisplaySize(free[0])));
		ret.append(String.format("%-35s%-15s%-15s%-15s\n", "Total mem (from cudaMemGetInfo)", 
				"-", "-", byteCountToDisplaySize(total[0])));
		ret.append("====================================================\n");
		return ret.toString();
	}
	
	/**
	 * Class that governs the eviction policy
	 */
	public static class EvictionPolicyBasedComparator implements Comparator<GPUObject> {
		public EvictionPolicyBasedComparator(long neededSize) {
		
		}
		
		@Override
		public int compare(GPUObject p1, GPUObject p2) {
			if (p1.isLocked() && p2.isLocked()) {
				// Both are locked, so don't sort
				return 0;
			} else if (p1.isLocked()) {
				// Put the unlocked one to RHS
				// a value less than 0 if x < y; and a value greater than 0 if x > y
				return -1;
			} else if (p2.isLocked()) {
				// Put the unlocked one to RHS
				// a value less than 0 if x < y; and a value greater than 0 if x > y
				return 1;
			} else {
				// Both are unlocked
				return Long.compare(p2.timestamp.get(), p1.timestamp.get());
			}
		}
	}
}
