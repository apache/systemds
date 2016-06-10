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
package org.apache.sysml.runtime.controlprogram.context;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class GPUContext {

	public static ArrayList<GPUObject> allocatedPointers = new ArrayList<GPUObject>(); 
	protected static GPUContext currContext;
	protected GPUContext() { }
	
	public static volatile Boolean isGPUContextCreated = false;
	
	public abstract long getAvailableMemory();
	
	// Creation / Destruction of GPUContext and related handles
	public static GPUContext createGPUContext() {
		if(currContext == null && DMLScript.USE_ACCELERATOR) {
			new Thread(new Runnable() {
				@Override
				public void run() {
					// Lazy GPU context creation
					synchronized(isGPUContextCreated) {
						currContext = new JCudaContext();
						OptimizerUtils.GPU_MEMORY_BUDGET = ((JCudaContext)currContext).getAvailableMemory();
						isGPUContextCreated = true;
					}
				}
			}).start();
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