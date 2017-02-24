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

import java.util.ArrayList;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

//FIXME merge JCudaContext into GPUContext as this context is anyway CUDA specific
public abstract class GPUContext {

	public static ArrayList<GPUObject> allocatedPointers = new ArrayList<GPUObject>();

	/** cudaFree calls are done asynchronously on a separate thread,
	 *  this list preserve the list of currently happening cudaFree calls */
	public static ConcurrentLinkedQueue<Future> pendingDeallocates = new ConcurrentLinkedQueue<Future>();

	/** All asynchronous cudaFree calls will be done on this executor service */
	public static ExecutorService deallocExecutorService = Executors.newSingleThreadExecutor();

	/** Synchronization object to make sure no allocations happen when something is being evicted from memory */
	public static final Object syncObj = new Object();

	protected static GPUContext currContext;
	public static volatile Boolean isGPUContextCreated = false;

	protected GPUContext() {}

	/**
	 * Gets device memory available for SystemML operations
	 * 
	 * @return available memory
	 */
	public abstract long getAvailableMemory();

	/**
	 * Ensures that all the CUDA cards on the current system are
	 * of the minimum required compute capability.
	 * (The minimum required compute capability is hard coded in {@link JCudaContext}.
	 * 
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public abstract void ensureComputeCapability() throws DMLRuntimeException;
	
	/**
	 * Singleton Factory method for creation of {@link GPUContext}
	 * @return GPU context
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static GPUContext getGPUContext() throws DMLRuntimeException {
		if(currContext == null && DMLScript.USE_ACCELERATOR) {
			synchronized(isGPUContextCreated) {
				currContext = new JCudaContext();
				currContext.ensureComputeCapability();
				OptimizerUtils.GPU_MEMORY_BUDGET = ((JCudaContext)currContext).getAvailableMemory();
				isGPUContextCreated = true;
			}
		}
		return currContext;
	}
	
	public static GPUObject createGPUObject(MatrixObject mo) {
		if(DMLScript.USE_ACCELERATOR) {
			synchronized(isGPUContextCreated) {
				if(currContext == null)
					throw new RuntimeException("GPUContext is not created");
				if(currContext instanceof JCudaContext)
					return new JCudaObject(mo);
			}
		}
		throw new RuntimeException("Cannot create createGPUObject when USE_ACCELERATOR is off");
	}
	public abstract void destroy() throws DMLRuntimeException;
	
	
}
