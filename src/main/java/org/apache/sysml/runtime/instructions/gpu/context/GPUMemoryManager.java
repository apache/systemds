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
package org.apache.sysml.runtime.instructions.gpu.context;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.JCuda.cudaMemset;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;
/**
 * - All cudaFree and cudaMalloc in SystemML should go through this class to avoid OOM or incorrect results.
 * - This class can be refactored in future to accept a chunk of memory ahead of time rather than while execution. This will only thow memory-related errors during startup.  
 */
public class GPUMemoryManager {
	protected static final Log LOG = LogFactory.getLog(GPUMemoryManager.class.getName());
	
	/*****************************************************************************************/
	// GPU Memory is divided into three major sections:
	// 1. Matrix Memory: Memory allocated to matrices in SystemML and addressable by GPUObjects.
	// This memory section is divided into three minor sections:
	// 1.1 Locked Matrix Memory
	// 1.2 UnLocked + Non-Dirty Matrix Memory
	// 1.3 UnLocked + Dirty Matrix Memory
	// To get the GPUObjects/Pointers in this section, please use getGPUObjects and getPointers methods of GPUMatrixMemoryManager.
	// To clear GPUObjects/Pointers in this section, please use clear and clearAll methods of GPUMatrixMemoryManager.
	// Both these methods allow to get/clear unlocked/locked and dirty/non-dirty objects of a certain size.
	GPUMatrixMemoryManager matrixMemoryManager;
	public GPUMatrixMemoryManager getGPUMatrixMemoryManager() {
		return matrixMemoryManager;
	}
	
	// 2. Rmvar-ed pointers: If sysml.gpu.eager.cudaFree is set to false,
	// then this manager caches pointers of the GPUObject on which rmvar instruction has been executed for future reuse.
	// We observe 2-3x improvement with this approach and hence recommend to set this flag to false.
	GPULazyCudaFreeMemoryManager lazyCudaFreeMemoryManager;
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
	HashMap<Pointer, PointerInfo> allPointers = new HashMap<>();
	
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
			if(DMLScript.PRINT_GPU_MEMORY_INFO) {
				this.stackTraceElements = Thread.currentThread().getStackTrace();
			}
			this.sizeInBytes = sizeInBytes;
		}
		public long getSizeInBytes() {
			return sizeInBytes;
		}
	}
	
	// If the available free size is less than this factor, GPUMemoryManager will warn users of multiple programs grabbing onto GPU memory.
	// This often happens if user tries to use both TF and SystemML, and TF grabs onto 90% of the memory ahead of time.
	private static final double WARN_UTILIZATION_FACTOR = 0.7;
	
	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application.
	public double GPU_MEMORY_UTILIZATION_FACTOR = ConfigurationManager.getDMLConfig()
			.getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
	
	
	public GPUMemoryManager(GPUContext gpuCtx) {
		matrixMemoryManager = new GPUMatrixMemoryManager(this);
		lazyCudaFreeMemoryManager = new GPULazyCudaFreeMemoryManager(this);
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
	 * @return allocated pointer
	 */
	private Pointer cudaMallocNoWarn(Pointer A, long size) {
		try {
			cudaMalloc(A, size);
			allPointers.put(A, new PointerInfo(size));
			return A;
		} catch(jcuda.CudaException e) {
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
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		long mallocStart = 0;
		// Step 1: First try reusing exact match in rmvarGPUPointers to avoid holes in the GPU memory
		Pointer A = lazyCudaFreeMemoryManager.getRmvarPointer(opcode, size);
		if(A != null)
			addMiscTime(opcode, GPUInstruction.MISC_TIMER_REUSE, t0);
		
		Pointer tmpA = (A == null) ? new Pointer() : null;
		// Step 2: Allocate a new pointer in the GPU memory (since memory is available)
		if(A == null && size <= getAvailableMemory()) {
			mallocStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
			A = cudaMallocNoWarn(tmpA, size); // Try malloc rather than check available memory to avoid fragmentation related issues
			addMiscTime(null, GPUStatistics.cudaEvictMallocTime, GPUStatistics.cudaEvictionMallocCount, GPUInstruction.MISC_TIMER_EVICT, mallocStart);
			if(LOG.isTraceEnabled()) {
				if(A == null)
					LOG.trace("Couldnot allocate a new pointer in the GPU memory:" + byteCountToDisplaySize(size));
				else
					LOG.trace("Allocated a new pointer in the GPU memory:" + byteCountToDisplaySize(size));
			}
		}
		
		// Reusing one rmvar-ed pointer (Step 3) is preferred to reusing multiple pointers as the latter may not be contiguously allocated.
		// (Step 4 or using any other policy that doesnot take memory into account).
		
		// Step 3: Try reusing non-exact match entry of rmvarGPUPointers
		if(A == null) { 
			A = lazyCudaFreeMemoryManager.getRmvarPointerMinSize(opcode, size);
			if(A != null) {
				guardedCudaFree(A);
				mallocStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
				A = cudaMallocNoWarn(tmpA, size); // Try malloc rather than check available memory to avoid fragmentation related issues
				addMiscTime(null, GPUStatistics.cudaEvictMallocTime, GPUStatistics.cudaEvictionMallocCount, GPUInstruction.MISC_TIMER_EVICT, mallocStart);
				if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
					if(A == null)
						LOG.info("Couldnot reuse non-exact match of rmvarGPUPointers:" + byteCountToDisplaySize(size));
					else {
						LOG.info("Reuses a non-exact match from rmvarGPUPointers:" + byteCountToDisplaySize(size));
						LOG.info("GPU Memory info after reusing a non-exact match from rmvarGPUPointers:" + toString());
					}
				}
			}
		}
		
		// Step 3.b: An optimization missing so as not to over-engineer malloc:
		// Try to find minimal number of contiguously allocated pointer.
		
		// Evictions of matrix blocks are expensive (as they might lead them to be written to disk in case of smaller CPU budget) 
		// than doing cuda free/malloc/memset. So, rmvar-ing every blocks (step 4) is preferred to eviction (step 5).
		
		// Step 4: Eagerly free-up rmvarGPUPointers and check if memory is available on GPU
		if(A == null) {
			lazyCudaFreeMemoryManager.clearAll();
			if(size <= getAvailableMemory()) {
				A = cudaMallocNoWarn(tmpA, size);
				if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
					if(A == null)
						LOG.info("Couldnot allocate a new pointer in the GPU memory after eager free:" + byteCountToDisplaySize(size));
					else {
						LOG.info("Allocated a new pointer in the GPU memory after eager free:" + byteCountToDisplaySize(size));
						LOG.info("GPU Memory info after allocating new pointer post lazyCudaFreeMemoryManager.clearAll():" + toString());
					}
				}
			}
		}
		
		addMiscTime(opcode, GPUStatistics.cudaAllocTime, GPUStatistics.cudaAllocCount, GPUInstruction.MISC_TIMER_ALLOCATE, t0);
		
		// Step 5: Try eviction based on the given policy
		if(A == null) {
			t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			
			// First, clear unlocked non-dirty matrices greater than or equal to size
			// Comparator clears the largest matrix to avoid future evictions
			boolean success = matrixMemoryManager.clear(false, false, size, SIMPLE_COMPARATOR_SORT_BY_SIZE, opcode);
			if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
				if(success) {
					LOG.info("Cleared an unlocked non-dirty matrix greater than or equal to " + byteCountToDisplaySize(size));
					LOG.info("GPU Memory info after clearing an unlocked non-dirty matrix:" + toString());
				}
				else
					LOG.info("No unlocked non-dirty matrix greater than or equal to " + byteCountToDisplaySize(size) + " found for clearing.");
			}
			if(!success) {
				// First, clear unlocked dirty matrices greater than or equal to size using the eviction policy
				// Comparator clears the largest matrix to avoid future evictions
				if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
					LOG.info("GPU Memory info before eviction:" + toString());
				}
				success = matrixMemoryManager.clear(false, true, size, new EvictionPolicyBasedComparator(size), opcode);
				// JCuda.cudaDeviceSynchronize();
				if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
					if(success) {
						LOG.info("Evicted an unlocked dirty matrix greater than or equal to " + byteCountToDisplaySize(size));
						LOG.info("GPU Memory info after evicting an unlocked dirty matrix:" + toString());
					}
					else
						LOG.info("No unlocked dirty matrix greater than or equal to " + byteCountToDisplaySize(size) + " found for evicted.");
				}
				
				if(!success) {
					// Minor optimization: clear all unlocked non-dirty matrices before attempting eviction 
					// Delete all non-dirty
					List<GPUObject> unlockedGPUObjects = matrixMemoryManager.gpuObjects.stream()
												.filter(gpuObj -> !gpuObj.isLocked() && !gpuObj.isDirty()).collect(Collectors.toList());
					matrixMemoryManager.gpuObjects.removeAll(unlockedGPUObjects);
					for(GPUObject toBeRemoved : unlockedGPUObjects) {
						toBeRemoved.clearData(opcode, true);
					}
					if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
						LOG.info("GPU Memory info after clearing all unlocked non-dirty matrices:" + toString());
					}
					mallocStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
					A = cudaMallocNoWarn(tmpA, size); // Try malloc rather than check available memory to avoid fragmentation related issues
					addMiscTime(null, GPUStatistics.cudaEvictMallocTime, GPUStatistics.cudaEvictionMallocCount, GPUInstruction.MISC_TIMER_EVICT, mallocStart);
				
					// ---------------------------------------------------------------
					// Evict unlocked GPU objects one-by-one and try malloc
					unlockedGPUObjects = null;
					if(A == null) {
						unlockedGPUObjects = matrixMemoryManager.gpuObjects.stream()
								.filter(gpuObj -> !gpuObj.isLocked() && gpuObj.isDirty()).collect(Collectors.toList());
						Collections.sort(unlockedGPUObjects, new EvictionPolicyBasedComparator(size));
						while(A == null && unlockedGPUObjects.size() > 0) {
							if(DMLScript.GPU_EVICTION_POLICY == DMLScript.EvictionPolicy.ALIGN_MEMORY) {
								// TODO: Optimize later using sliding window
								// Evict as many sequential dense objects from back of the queue as possible
								long neededSize = size;
								while(neededSize >= 0 && unlockedGPUObjects.size() > 0) {
									GPUObject gpuObj = unlockedGPUObjects.remove(unlockedGPUObjects.size()-1);
									neededSize -= matrixMemoryManager.getWorstCaseContiguousMemorySize(gpuObj);
									gpuObj.copyFromDeviceToHost(opcode, true, true);
								}
							}
							else {
								GPUObject gpuObj = unlockedGPUObjects.remove(unlockedGPUObjects.size()-1);
								gpuObj.copyFromDeviceToHost(opcode, true, true);
							}
							mallocStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
							A = cudaMallocNoWarn(tmpA, size); // Try malloc rather than check available memory to avoid fragmentation related issues
							addMiscTime(null, GPUStatistics.cudaEvictMallocTime, GPUStatistics.cudaEvictionMallocCount, GPUInstruction.MISC_TIMER_EVICT, mallocStart);
						}
						if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
							// greater than or equal to " + byteCountToDisplaySize(size)
							LOG.info("GPU Memory info after eviction:" + toString());
						}
						if(unlockedGPUObjects != null && unlockedGPUObjects.size() == 0) {
							LOG.warn("Evicted all unlocked matrices");
						}
					}
					
				}
				// ---------------------------------------------------------------
			}
			addMiscTime(opcode, GPUStatistics.cudaEvictTime, GPUStatistics.cudaEvictionCount, GPUInstruction.MISC_TIMER_EVICT, t0);
			if(A == null) {
				A = cudaMallocNoWarn(tmpA, size); // if the matrix is not allocated via eviction
			}
			if(A == null) {
				LOG.warn("cudaMalloc failed immediately after cudaMemGetInfo reported that memory of size " 
						+ byteCountToDisplaySize(size) + " is available. "
						+ "This usually happens if there are external programs trying to grab on to memory in parallel or there is potential fragmentation.");
			}
			else if(DMLScript.PRINT_GPU_MEMORY_INFO || LOG.isTraceEnabled()) {
				LOG.info("Malloc after eviction/clearing is successful.");
			}
		}
		
		// Step 6: Handle defragmentation
		if(A == null) {
			LOG.warn("Potential fragmentation of the GPU memory. Forcibly evicting all ...");
			LOG.info("Before clearAllUnlocked, GPU Memory info:" + toString());
			matrixMemoryManager.clearAllUnlocked(opcode);
			LOG.info("GPU Memory info after evicting all unlocked matrices:" + toString());
			A = cudaMallocNoWarn(tmpA, size);
		}
		
		if(A == null) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix, requested = " + byteCountToDisplaySize(size) + ". \n "
					+ toString());
		}
		
		t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		cudaMemset(A, 0, size);
		addMiscTime(opcode, GPUStatistics.cudaMemSet0Time, GPUStatistics.cudaMemSet0Count, GPUInstruction.MISC_TIMER_SET_ZERO, t0);
		return A;
	}
	
	// --------------- Developer Utilities to debug potential memory leaks ------------------------
	@SuppressWarnings("unused")
	private void printPointers(List<PointerInfo> pointers) {
		for(PointerInfo ptrInfo : pointers) {
			System.out.println(">>" + 
					// getCallerInfo(ptrInfo.stackTraceElements, 5) + getCallerInfo(ptrInfo.stackTraceElements, 6) + getCallerInfo(ptrInfo.stackTraceElements, 7) +
					getCallerInfo(ptrInfo.stackTraceElements, 8) + getCallerInfo(ptrInfo.stackTraceElements, 9) + getCallerInfo(ptrInfo.stackTraceElements, 10));
		}
	}
	
	@SuppressWarnings("unused")
	private void printPointers(Set<Pointer> pointers, StringBuilder sb) {
		for(Pointer ptr : pointers) {
			PointerInfo ptrInfo = allPointers.get(ptr);
			sb.append(">>");
			// getCallerInfo(ptrInfo.stackTraceElements, 5) + getCallerInfo(ptrInfo.stackTraceElements, 6) + getCallerInfo(ptrInfo.stackTraceElements, 7) +
			sb.append(getCallerInfo(ptrInfo.stackTraceElements, 8));
			sb.append(getCallerInfo(ptrInfo.stackTraceElements, 9));
			sb.append(getCallerInfo(ptrInfo.stackTraceElements, 10));
			sb.append("\n");
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
			cudaFree(toFree);
			// JCuda.cudaDeviceSynchronize(); // Force a device synchronize after free-ing the pointer for debugging
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
		matrixMemoryManager.gpuObjects.removeIf(a -> a.equals(gpuObj));
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
			if (opcode != null && DMLScript.FINEGRAINED_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(opcode, instructionLevelTimer, totalTime);
		}
	}
	
	/**
	 * Convenient method to add misc timers
	 * 
	 * @param opcode opcode
	 * @param instructionLevelTimer member of GPUInstruction
	 * @param startTime start time
	 */
	void addMiscTime(String opcode, String instructionLevelTimer, long startTime) {
		if (opcode != null && DMLScript.FINEGRAINED_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(opcode, instructionLevelTimer, System.nanoTime() - startTime);
	}
	
	
	/**
	 * Print debugging information
	 */
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
		//if(DMLScript.PRINT_GPU_MEMORY_INFO) {
		//	if(potentiallyLeakyPointers.size() > 0) {
		//		ret.append("Non-matrix pointers were allocated by:\n");
		//		printPointers(potentiallyLeakyPointers, ret);
		//	}
		//}
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
	 * Gets the available memory on GPU that SystemML can use.
	 *
	 * @return the available memory in bytes
	 */
	public long getAvailableMemory() {
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		return (long) (free[0] * GPU_MEMORY_UTILIZATION_FACTOR);
	}
	
	private static Comparator<GPUObject> SIMPLE_COMPARATOR_SORT_BY_SIZE = (o1, o2) -> o1.getSizeOnDevice() < o2.getSizeOnDevice() ? -1 : 1;
	
	private static class CustomPointer extends Pointer {
		public CustomPointer(Pointer p) {
			super(p);
		}
		
		@Override
		public long getNativePointer() {
			return super.getNativePointer();
		}
		
	}
	/**
	 * Class that governs the eviction policy
	 */
	public static class EvictionPolicyBasedComparator implements Comparator<GPUObject> {
		private long neededSize;
		public EvictionPolicyBasedComparator(long neededSize) {
			this.neededSize = neededSize;
		}
		
		private int minEvictCompare(GPUObject p1, GPUObject p2) {
			long p1Size = p1.getSizeOnDevice() - neededSize;
			long p2Size = p2.getSizeOnDevice() - neededSize;

			if (p1Size >= 0 && p2Size >= 0) {
				return Long.compare(p2Size, p1Size);
			} else {
				return Long.compare(p1Size, p2Size);
			}
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
				if (DMLScript.GPU_EVICTION_POLICY == DMLScript.EvictionPolicy.ALIGN_MEMORY) {
					if(!p1.isDensePointerNull() && !p2.isDensePointerNull()) {
						long p1Ptr = new CustomPointer(p1.getDensePointer()).getNativePointer();
						long p2Ptr = new CustomPointer(p2.getDensePointer()).getNativePointer();
						
						if(p1Ptr <= p2Ptr)
							return -1;
						else
							return 1;
					}
					else if(p1.isDensePointerNull() && !p2.isDensePointerNull()) {
						return -1;
					}
					else if(!p1.isDensePointerNull() && p2.isDensePointerNull()) {
						return 1;
					}
					else {
						return minEvictCompare(p1, p2);
					}
				}
				else if (DMLScript.GPU_EVICTION_POLICY == DMLScript.EvictionPolicy.MIN_EVICT) {
					return minEvictCompare(p1, p2);
				} else {
					return Long.compare(p2.timestamp.get(), p1.timestamp.get());
				}
			}
		}
	}
}
