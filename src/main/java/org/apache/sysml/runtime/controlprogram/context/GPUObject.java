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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class GPUObject {

	public boolean isDeviceCopyModified = false;
	volatile boolean isLocked = false;
	public int numReferences = 1;
	
	MatrixObject mat = null;
	protected GPUObject(MatrixObject mat2)  {
		this.mat = mat2;
	}
	
	// package-level visibility as these methods are guarded by underlying GPUContext
	abstract void allocateMemoryOnDevice() throws DMLRuntimeException;
	abstract void deallocateMemoryOnDevice() throws DMLRuntimeException;
	abstract long getSizeOnDevice() throws DMLRuntimeException;
	abstract void copyFromHostToDevice() throws DMLRuntimeException;
	abstract void copyFromDeviceToHost() throws DMLRuntimeException; // Called by export()
	
	static GPUObject createGPUObject(MatrixObject mat2, GPUContext gpuCtx) throws DMLRuntimeException {
		if(gpuCtx instanceof JCudaContext) {
			return new JCudaObject(mat2);
		}
		throw new DMLRuntimeException("Unsupported GPUContext");
	}
}