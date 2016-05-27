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

import java.util.Collections;
import java.util.Comparator;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class GPUObject {

	public boolean isDeviceCopyModified = false;
	volatile boolean isLocked = false;
	
	MatrixObject mat = null;
	protected GPUObject(MatrixObject mat2)  {
		this.mat = mat2;
	}
	
	public abstract void acquireDeviceRead() throws DMLRuntimeException;
	public abstract void acquireDeviceModify() throws DMLRuntimeException;
	public abstract void acquireHostRead() throws CacheException;
	public abstract void acquireHostModify() throws CacheException;
	public abstract void release(boolean isGPUCopyModified) throws CacheException;
	
	
	// package-level visibility as these methods are guarded by underlying GPUContext
	abstract void allocateMemoryOnDevice() throws DMLRuntimeException;
	abstract void deallocateMemoryOnDevice() throws DMLRuntimeException;
	abstract long getSizeOnDevice() throws DMLRuntimeException;
	abstract void copyFromHostToDevice() throws DMLRuntimeException;
	abstract void copyFromDeviceToHost() throws DMLRuntimeException; // Called by export()
	
	
	/**
	 * It finds matrix toBeRemoved such that toBeRemoved.GPUSize >= size
	 * // TODO: it is the smallest matrix size that satisfy the above condition. For now just evicting the largest pointer.
	 * Then returns toBeRemoved. 
	 * 
	 */
	protected void evict(long GPUSize) throws DMLRuntimeException {
		if(GPUContext.allocatedPointers.size() == 0) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
		}
		
		synchronized(evictionLock) {
			Collections.sort(GPUContext.allocatedPointers, new Comparator<GPUObject>() {
	
				@Override
				public int compare(GPUObject p1, GPUObject p2) {
					if(p1.isLocked && p2.isLocked) {
						return 0;
					}
					else if(p1.isLocked && !p2.isLocked) {
						// p2 by default is considered larger
						return 1;
					}
					else if(!p1.isLocked && p2.isLocked) {
						return -1;
					}
					long p1Size = 0; long p2Size = 0;
					try {
						p1Size = p1.getSizeOnDevice();
						p2Size = p2.getSizeOnDevice();
					} catch (DMLRuntimeException e) {
						throw new RuntimeException(e);
					}
					if(p1Size == p2Size) {
						return 0;
					}
					else if(p1Size < p2Size) {
						return 1;
					}
					else {
						return -1;
					}
				}
			});
			
			
			while(GPUSize > getAvailableMemory() && GPUContext.allocatedPointers.size() > 0) {
				GPUObject toBeRemoved = GPUContext.allocatedPointers.get(GPUContext.allocatedPointers.size() - 1);
				if(toBeRemoved.isLocked) {
					throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
				}
				if(toBeRemoved.isDeviceCopyModified) {
					toBeRemoved.copyFromDeviceToHost();
				}
				toBeRemoved.clearData();
			}
		}
	}
	
	public void clearData() throws CacheException {
		synchronized(evictionLock) {
			GPUContext.allocatedPointers.remove(this);
		}
		try {
			deallocateMemoryOnDevice();
		} catch (DMLRuntimeException e) {
			throw new CacheException(e);
		}
	}
	
	static Boolean evictionLock = new Boolean(true);
	
	protected long getAvailableMemory() {
		return GPUContext.currContext.getAvailableMemory();
	}
	
//	// Copying from device -> host occurs here
//	// Called by MatrixObject's exportData
//	public void exportData() throws CacheException {
//		boolean isDeviceCopyModified = mat.getGPUObject() != null && mat.getGPUObject().isDeviceCopyModified;
//		boolean isHostCopyUnavailable = mat.getMatrixBlock() == null || 
//				(mat.getMatrixBlock().getDenseBlock() == null && mat.getMatrixBlock().getSparseBlock() == null);
//		
//		if(mat.getGPUObject() != null && (isDeviceCopyModified || isHostCopyUnavailable)) {
//			try {
//				mat.getGPUObject().copyFromDeviceToHost();
//			} catch (DMLRuntimeException e) {
//				throw new CacheException(e);
//			}
//		}
//	}
}