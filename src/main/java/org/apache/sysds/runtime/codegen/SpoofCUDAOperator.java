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

package org.apache.sysds.runtime.codegen;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import jcuda.Pointer;

import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;

import static jcuda.runtime.cudaError.cudaSuccess;
import static org.apache.sysds.runtime.matrix.data.LibMatrixCUDA.sizeOfDataType;

public interface SpoofCUDAOperator  {
	// these two constants have equivalences in native code:
	int JNI_MAT_ENTRY_SIZE = 40;
	int TRANSFERRED_DATA_HEADER_SIZE = 32;

	abstract class PrecisionProxy {
		protected final long ctx;
		
		public PrecisionProxy() { ctx = SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA);	}
		
		public abstract int exec(SpoofCUDAOperator op);
	}
	
	String getName();

	default void writeMatrixDescriptorToBuffer(ByteBuffer dst, int rows, int cols, long row_ptr,
		long col_idx_ptr, long data_ptr, long nnz)
	{
		dst.putLong(nnz);
		dst.putInt(rows);
		dst.putInt(cols);
		dst.putLong(row_ptr);
		dst.putLong(col_idx_ptr);
		dst.putLong(data_ptr);
	}

	default void prepareMatrixPointers(ByteBuffer buf, ExecutionContext ec, MatrixObject mo, boolean tB1) {
		if(mo.getGPUObject(ec.getGPUContext(0)).isSparse()) {
			writeMatrixDescriptorToBuffer(buf, (int)mo.getNumRows(), (int)mo.getNumColumns(),
				GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(mo).rowPtr),
					GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(mo).colInd),
						GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(mo).val),
							ec.getGPUSparsePointerAddress(mo).nnz);
		}
		else {
			if(tB1) {
				int rows = (int)mo.getNumRows();
				int cols = (int)mo.getNumColumns();
				Pointer b1 = mo.getGPUObject(ec.getGPUContext(0)).getDensePointer();
				Pointer ptr = ec.getGPUContext(0).allocate(getName(), (long) rows * cols * sizeOfDataType, false);
				LibMatrixCUDA.denseTranspose(ec, ec.getGPUContext(0), getName(), b1, ptr, rows, cols);
				writeMatrixDescriptorToBuffer(buf, rows, cols, 0, 0, GPUObject.getPointerAddress(ptr), mo.getNnz());
			} else {
				writeMatrixDescriptorToBuffer(buf, (int)mo.getNumRows(), (int)mo.getNumColumns(), 0, 0,
					ec.getGPUDensePointerAddress(mo), mo.getNnz());
			}
		}
	}

	default void packDataForTransfer(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects, MatrixObject out_obj, int num_inputs, int ID, long grix, boolean tB1,
			Pointer[] ptr)
	{
		int op_data_size = (inputs.size() + 1) * JNI_MAT_ENTRY_SIZE + scalarObjects.size() * Double.BYTES + TRANSFERRED_DATA_HEADER_SIZE;
		Pointer staging = new Pointer();
		if(SpoofOperator.getNativeStagingBuffer(staging, this.getContext(), op_data_size) != cudaSuccess)
			throw new RuntimeException("Failed to get native staging buffer from spoof operator");
		ByteBuffer buf = staging.getByteBuffer();
		buf.putInt(op_data_size);
		buf.putInt(ID);
		buf.putInt((int)grix);
		buf.putInt(num_inputs);
		buf.putInt(inputs.size() - num_inputs);
		buf.putInt(out_obj == null ? 0 : 1);
		buf.putInt(scalarObjects.size());
		buf.putInt(-1); // padding

		// copy input & side input pointers
		for(int i=0; i < inputs.size(); i++) {
			if(i == num_inputs)
				prepareMatrixPointers(buf, ec, inputs.get(i), tB1);
			else
				prepareMatrixPointers(buf, ec, inputs.get(i), false);
		}

		// copy output pointers or allocate buffer for reduction
 		if(out_obj == null) {
			long num_blocks = 1;
			if(this instanceof SpoofCUDACellwise) {
				int NT = 256;
				long N = inputs.get(0).getNumRows() * inputs.get(0).getNumColumns();
				num_blocks = ((N + NT * 2 - 1) / (NT * 2));
				ptr[0] = ec.getGPUContext(0).allocate(getName(), LibMatrixCUDA.sizeOfDataType * num_blocks, false);
			}
			else
				ptr[0] = ec.getGPUContext(0).allocate(getName(), LibMatrixCUDA.sizeOfDataType * num_blocks, true);
			writeMatrixDescriptorToBuffer(buf, 1, 1, 0, 0, GPUObject.getPointerAddress(ptr[0]), 1);
		}
 		else {
			prepareMatrixPointers(buf, ec, out_obj, false);
		}

 		// copy scalar values (no pointers)
		for(ScalarObject scalarObject : scalarObjects) {
			buf.putDouble(scalarObject.getDoubleValue());
		}
	}

	MatrixObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
			ArrayList<ScalarObject> scalarObjects, String outputName);
	
	ScalarObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects);

	int execute_dp(long ctx);
	int execute_sp(long ctx);
	long getContext();
}
