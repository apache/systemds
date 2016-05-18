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
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class GPUContext {

	protected ArrayList<GPUObject> allocatedPointers = new ArrayList<GPUObject>(); 
	protected static GPUContext currContext;
	protected GPUContext() { }
	
	// Creation / Destruction of GPUContext and related handles
	public static GPUContext createGPUContext() {
		if(currContext == null && DMLScript.USE_ACCELERATOR)
			currContext = new JCudaContext();
		return currContext;
	}
	public abstract void destroy() throws DMLRuntimeException;
	
	// Bufferpool-related methods
	abstract void acquireRead(MatrixObject mat) throws DMLRuntimeException;
	abstract void acquireModify(MatrixObject mat) throws DMLRuntimeException;
	abstract void release(MatrixObject mat, boolean isGPUCopyModified);
	public abstract void remove(MatrixObject mat) throws DMLRuntimeException;
	
	// Copying from device -> host occurs here
	// Called by MatrixObject's exportData
	public static void exportData(MatrixObject mo) throws CacheException {
		if(currContext == null) {
			throw new CacheException("GPUContext is not initialized");
		}
		boolean isDeviceCopyModified = mo.getGPUObject() != null && mo.getGPUObject().isDeviceCopyModified;
		boolean isHostCopyUnavailable = mo.getMatrixBlock() == null || 
				(mo.getMatrixBlock().getDenseBlock() == null && mo.getMatrixBlock().getSparseBlock() == null);
		
		if(mo.getGPUObject() != null && (isDeviceCopyModified || isHostCopyUnavailable)) {
			try {
				mo.getGPUObject().copyFromDeviceToHost();
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
	}
	
}