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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

//FIXME merge JCudaContext into GPUContext as this context is anyway CUDA specific

@SuppressWarnings("rawtypes")
public abstract class GPUContext {

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
				OptimizerUtils.GPU_MEMORY_BUDGET = currContext.getAvailableMemory();
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
