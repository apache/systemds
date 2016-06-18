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

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import jcuda.Pointer;
import jcuda.Sizeof;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.utils.Statistics;

public class JCudaObject extends GPUObject {
	
	public Pointer jcudaPointer = null;

	JCudaObject(MatrixObject mat2) {
		super(mat2);
	}
	
	private void prepare(boolean isInput, int numElemsToAllocate) throws DMLRuntimeException {
		if(jcudaPointer != null) {
			// Already allocated on GPU and expected to be in sync
			// checkDimensions();
		}
		else {
			long GPUSize;
			if(numElemsToAllocate != -1)
				GPUSize = (Sizeof.DOUBLE) * (long) (numElemsToAllocate);
			else
				GPUSize = getSizeOnDevice();
			// Ensure enough memory while allocating the matrix
			if(GPUSize > getAvailableMemory()) {
				evict(GPUSize);
			}
			allocateMemoryOnDevice(numElemsToAllocate);
			synchronized(evictionLock) {
				GPUContext.allocatedPointers.add(this);
			}
			if(isInput)
				copyFromHostToDevice();
		}
		numLocks.addAndGet(1);
	}
	
	@Override
	public void acquireDeviceRead() throws DMLRuntimeException {
		prepare(true, -1);
		if(!isAllocated) 
			throw new DMLRuntimeException("Expected device data to be allocated");
	}
	
//	private void checkDimensions() throws DMLRuntimeException {
//		if(LibMatrixCUDA.isInSparseFormat(mat))
//			throw new DMLRuntimeException("Sparse format not implemented");
//		else {
//			if(mat.getNumRows()*mat.getNumColumns() != numElems) {
//				throw new DMLRuntimeException("The jcudaPointer and MatrixBlock is not in synched");
//			}
//		}
//	}
	
	@Override
	public void acquireDenseDeviceModify(int numElemsToAllocate) throws DMLRuntimeException {
		prepare(false, numElemsToAllocate); 
		isDeviceCopyModified = true;
		if(!isAllocated) 
			throw new DMLRuntimeException("Expected device data to be allocated");
	}
	
	@Override
	public void acquireHostRead() throws CacheException {
		if(isAllocated) {
			try {
				if(isDeviceCopyModified) {
					copyFromDeviceToHost();
				}
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
	}
	
	@Override
	public void acquireHostModify() throws CacheException {
		if(isAllocated) {
			try {
				if(isDeviceCopyModified) {
					throw new DMLRuntimeException("Potential overwrite of GPU data");
					// copyFromDeviceToHost();
				}
				clearData();
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
	}
	
	public void release(boolean isGPUCopyModified) throws CacheException {
		if(numLocks.addAndGet(-1) < 0) {
			throw new CacheException("Redundant release of GPU object");
		}
		isDeviceCopyModified = isGPUCopyModified;
	}

	@Override
	void allocateMemoryOnDevice(int numElemToAllocate) throws DMLRuntimeException {
		if(jcudaPointer == null) {
			long start = System.nanoTime();
			jcudaPointer = new Pointer();
			if(numElemToAllocate == -1 && LibMatrixCUDA.isInSparseFormat(mat))
				throw new DMLRuntimeException("Sparse format not implemented");
			else if(numElemToAllocate == -1) {
				// Called for dense input
				cudaMalloc(jcudaPointer,  mat.getNumRows()*mat.getNumColumns()*Sizeof.DOUBLE);
			}
			else {
				// Called for dense output
				cudaMalloc(jcudaPointer,  numElemToAllocate*Sizeof.DOUBLE);
			}
			
			Statistics.cudaAllocTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaAllocCount.addAndGet(1);
		}
		isAllocated = true;
	}
	
	@Override
	void deallocateMemoryOnDevice() {
		if(jcudaPointer != null) {
			long start = System.nanoTime();
			cudaFree(jcudaPointer);
			Statistics.cudaDeAllocTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaDeAllocCount.addAndGet(1);
		}
		jcudaPointer = null;
		isAllocated = false;
		numLocks.set(0);
	}
	
	@Override
	void copyFromHostToDevice() 
		throws DMLRuntimeException 
	{
		if( jcudaPointer == null )
			throw new DMLRuntimeException("Cannot copy from host to device without allocating");
		if( LibMatrixCUDA.isInSparseFormat(mat) )
			throw new DMLRuntimeException("Sparse format not implemented");
		
		printCaller();
		long start = System.nanoTime();
		
		MatrixBlock tmp = mat.acquireRead();
		double[] data = tmp.getDenseBlock();
		
		if( data == null && tmp.getSparseBlock() != null )
			throw new DMLRuntimeException("Incorrect sparsity calculation");
		else if( data==null && tmp.getNonZeros() != 0 )
			throw new DMLRuntimeException("MatrixBlock is not allocated");
		else if( tmp.getNonZeros() == 0 )
			data = new double[tmp.getNumRows()*tmp.getNumColumns()];
		
		cudaMemcpy(jcudaPointer, Pointer.to(data), mat.getNumRows()*mat.getNumColumns() * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
		mat.release();
		
		Statistics.cudaToDevTime.addAndGet(System.nanoTime()-start);
		Statistics.cudaToDevCount.addAndGet(1);
	}

	@Override
	protected void copyFromDeviceToHost() throws DMLRuntimeException {
		if(jcudaPointer != null) {
			printCaller();
			if(LibMatrixCUDA.isInSparseFormat(mat))
				throw new DMLRuntimeException("Sparse format not implemented");
			else {
				long start = System.nanoTime();
				MatrixBlock tmp = new MatrixBlock((int)mat.getNumRows(), (int)mat.getNumColumns(), false);
				tmp.allocateDenseBlock();
				double [] data = tmp.getDenseBlock();
				
				cudaMemcpy(Pointer.to(data), jcudaPointer, data.length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
				
				tmp.recomputeNonZeros();
				mat.acquireModify(tmp);
				mat.release();
				
				Statistics.cudaFromDevTime.addAndGet(System.nanoTime()-start);
				Statistics.cudaFromDevCount.addAndGet(1);
			}
		}
		else {
			throw new DMLRuntimeException("Cannot copy from device to host as JCuda pointer is not allocated");
		}
		isDeviceCopyModified = false;
	}

	@Override
	protected long getSizeOnDevice() throws DMLRuntimeException {
		long GPUSize = 0;
		int rlen = (int) mat.getNumRows();
		int clen = (int) mat.getNumColumns();

		if(LibMatrixCUDA.isInSparseFormat(mat)) {
			throw new DMLRuntimeException("Sparse format not implemented");
		}
		else {
			GPUSize = (Sizeof.DOUBLE) * (long) (rlen * clen);
		}
		return GPUSize;
	}
	
	private String getClassAndMethod(StackTraceElement st) {
		String [] str = st.getClassName().split("\\.");
		return str[str.length - 1] + "." + st.getMethodName();
	}
	
	private void printCaller() {
		if(JCudaContext.DEBUG) {
			StackTraceElement[] st = Thread.currentThread().getStackTrace();
			String ret = getClassAndMethod(st[1]);
			for(int i = 2; i < st.length && i < 7; i++) {
				ret += "->" + getClassAndMethod(st[i]);
			}
			System.out.println("CALL_STACK:" + ret);
		}
			
	}
}