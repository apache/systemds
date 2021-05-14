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

import jcuda.Pointer;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;

import java.util.ArrayList;

import static org.apache.sysds.runtime.matrix.data.LibMatrixCUDA.sizeOfDataType;

public interface SpoofCUDAOperator  {
	int JNI_MAT_ENTRY_SIZE = 6;
	abstract class PrecisionProxy {
		protected final long ctx;
		
		public PrecisionProxy() { ctx = SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA);	}
		
		public abstract int exec(ExecutionContext ec, SpoofCUDAOperator op, int opID, long[] in, long[] sides, long[] out,
				ArrayList<ScalarObject> scalarObjects, long grix);
		
		protected Pointer transferScalars(ExecutionContext ec, SpoofCUDAOperator op, int sizeOfDataType,
				ArrayList<ScalarObject> scalarObjects) {
			double[] s = SpoofOperator.prepInputScalars(scalarObjects);
			Pointer ptr = ec.getGPUContext(0).allocate(op.getName(), (long) scalarObjects.size() * sizeOfDataType);
			LibMatrixCUDA.cudaSupportFunctions.hostToDevice(ec.getGPUContext(0), s, ptr, op.getName());
			return ptr;
		}
	}
	
	String getName();
	
	void setScalarPtr(Pointer ptr);
	
	Pointer getScalarPtr();
	
	void releaseScalarGPUMemory(ExecutionContext ec);
	
	default long [] prepareInputPointers(ExecutionContext ec, ArrayList<MatrixObject> inputs, int offset) {
		long [] in = new long[offset * JNI_MAT_ENTRY_SIZE];
		for(int i = 0; i < offset; i++) {
			int j = i  * JNI_MAT_ENTRY_SIZE;
			
			if(inputs.get(i).getGPUObject(ec.getGPUContext(0)).isSparse()) {
				in[j] = ec.getGPUSparsePointerAddress(inputs.get(i)).nnz;
				in[j + 1] = inputs.get(i).getNumRows();
				in[j + 2] = inputs.get(i).getNumColumns();
				in[j + 3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).rowPtr);
				in[j + 4] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).colInd);
				in[j + 5] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).val);
			}
			else {
				in[j] = inputs.get(i).getNnz();
				in[j + 1] = inputs.get(i).getNumRows();
				in[j + 2] = inputs.get(i).getNumColumns();
				in[j + 5] = ec.getGPUDensePointerAddress(inputs.get(i));
			}
		}
		return in;
	}
	
	default long [] prepareSideInputPointers(ExecutionContext ec, ArrayList<MatrixObject> inputs, int offset, boolean tB1) {
		long[] sides = new long[(inputs.size() - offset) * JNI_MAT_ENTRY_SIZE];
		for(int i = offset; i < inputs.size(); i++) {
			int j = (i - offset)  * JNI_MAT_ENTRY_SIZE;
			if(inputs.get(i).getGPUObject(ec.getGPUContext(0)).isSparse()) {
				sides[j] = ec.getGPUSparsePointerAddress(inputs.get(i)).nnz;
				sides[j + 1] = inputs.get(i).getNumRows();
				sides[j + 2] = inputs.get(i).getNumColumns();
				sides[j + 3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).rowPtr);
				sides[j + 4] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).colInd);
				sides[j + 5] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).val);
			}
			else {
				if(tB1 && j == 0) {
					long rows = inputs.get(i).getNumRows();
					long cols = inputs.get(i).getNumColumns();
					Pointer b1 = inputs.get(i).getGPUObject(ec.getGPUContext(0)).getDensePointer();
					Pointer ptr = ec.getGPUContext(0).allocate(getName(), rows * cols * sizeOfDataType);
					
//					double[] tmp1 = new double[(int) (rows * cols)];
//					LibMatrixCUDA.cudaSupportFunctions.deviceToHost(ec.getGPUContext(0), b1, tmp1, getName(), false);
//
//					System.out.println("Mat before transpose: rows=" + rows + " cols=" + cols + "\n");
//					for(int m = 0; m < rows; m++) {
//						StringBuilder sb = new StringBuilder();
//						for(int n = 0; n < cols; n++)
//							sb.append(" " + tmp1[(int) (cols * m + n)]);
//						System.out.println(sb.toString());
//					}
					
					LibMatrixCUDA.denseTranspose(ec, ec.getGPUContext(0), getName(),
						b1, ptr, rows, cols);
					
//					double[] tmp2 = new double[(int) (rows * cols)];
//					LibMatrixCUDA.cudaSupportFunctions.deviceToHost(ec.getGPUContext(0), ptr, tmp2, getName(), false);
//
//					System.out.println("Mat after transpose: rows=" + cols + " cols=" + rows + "\n");
//					for(int m = 0; m < cols; m++) {
//						StringBuilder sb = new StringBuilder();
//						for(int n = 0; n < rows; n++)
//							sb.append(" " + tmp2[(int) (rows * m + n)]);
//						System.out.println(sb.toString());
//					}

					sides[j] = inputs.get(i).getNnz();
					sides[j + 1] = cols;
					sides[j + 2] = rows;
					sides[j + 5] = GPUObject.getPointerAddress(ptr);
					
				} else {
					sides[j] = inputs.get(i).getNnz();
					sides[j + 1] = inputs.get(i).getNumRows();
					sides[j + 2] = inputs.get(i).getNumColumns();
					sides[j + 5] = ec.getGPUDensePointerAddress(inputs.get(i));
				}
			}
		}
		return sides;
	}
	
	default long[] prepareOutputPointers(ExecutionContext ec, MatrixObject output, boolean sparseOut) {
		long[] out = {0,0,0,0,0,0};

		if(sparseOut) {
			out[0] = ec.getGPUSparsePointerAddress(output).nnz;
			out[1] = output.getNumRows();
			out[2] = output.getNumColumns();
			out[3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(output).rowPtr);
			out[4] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(output).colInd);
			out[5] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(output).val);
		}
		else {
			out[0] = output.getNnz();
			out[1] = output.getNumRows();
			out[2] = output.getNumColumns();
			out[5] = ec.getGPUDensePointerAddress(output);
		}
		return out;
	}
	
	MatrixObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs, 
			ArrayList<ScalarObject> scalarObjects, String outputName);
	
	ScalarObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects);
	
	int execute_sp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
	int execute_dp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
}
