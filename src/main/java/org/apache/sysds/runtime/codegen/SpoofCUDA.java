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

import java.util.ArrayList;

import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysds.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class SpoofCUDA extends SpoofOperator {
	private static final long serialVersionUID = -2161276866245388359L;
	
	private final CNodeTpl cnt;
	public final String name;
	public final String src;

	public SpoofCUDA(CNodeTpl cnode, String source) {
		name = "codegen." + cnode.getVarname();
		cnt = cnode;
		src = source;
	}

	public String getName() {
		return name;
	}

	public CNodeTpl getCNodeTemplate() {
		return cnt;
	}

	public String getSpoofTemplateType() {
		if (cnt instanceof CNodeCell)
			return "CW";
		else if(cnt instanceof CNodeRow)
			return "RA";
		else if(cnt instanceof CNodeMultiAgg)
			return "MA";
		else if(cnt instanceof CNodeOuterProduct)
			return "OP";
		else
			throw new RuntimeException("unknown spoof operator type");
	}
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
		throw new RuntimeException("method not implemented for SpoofNativeCUDA");
	}

	public double execute(ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects, MatrixObject out_obj,
							   ExecutionContext ec, boolean sparseOut) {
		double ret;
		long[] out_ptr = {0,0,0,0};
		
		if(out_obj != null) {
			if(sparseOut) {
				out_ptr[0] = ec.getGPUSparsePointerAddress(out_obj).nnz;
				out_ptr[1] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(out_obj).rowPtr);
				out_ptr[2] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(out_obj).colInd);
				out_ptr[3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(out_obj).val);
			}
			else
				out_ptr[3] = ec.getGPUDensePointerAddress(out_obj);
		}
		
		int offset = 1;
		if(cnt instanceof CNodeOuterProduct)
			offset = 2;
		
		// only dense input preparation for now
		long[] in_ptrs = new long[offset * 4];
		for(int i = 0; i < offset; i += 4) {
			if(inputs.get(i).getGPUObject(ec.getGPUContext(0)).isSparse()) {
				in_ptrs[i * 4] = ec.getGPUSparsePointerAddress(inputs.get(i)).nnz;
				in_ptrs[i * 4 + 1] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).rowPtr);
				in_ptrs[i * 4 + 2] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).colInd);
				in_ptrs[i * 4 + 3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).val);
			}
			else
				in_ptrs[i * 4 + 3] = ec.getGPUDensePointerAddress(inputs.get(i));
		}
		
		long[] side_ptrs = new long[(inputs.size() - offset) * 4];
		for(int i = offset; i < inputs.size(); i += 4) {
			int j = (i - offset)  * 4;
			if(inputs.get(i).getGPUObject(ec.getGPUContext(0)).isSparse()) {
				side_ptrs[j] = ec.getGPUSparsePointerAddress(inputs.get(i)).nnz;
				side_ptrs[j + 1] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).rowPtr);
				side_ptrs[j + 2] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).colInd);
				side_ptrs[j + 3] = GPUObject.getPointerAddress(ec.getGPUSparsePointerAddress(inputs.get(i)).val);
			}
			else
				side_ptrs[j + 3] = ec.getGPUDensePointerAddress(inputs.get(i));
		}


//		long[] side_ptrs = new long[inputs.size() - offset];
//		for(int i = offset; i < inputs.size(); ++i)
//			side_ptrs[i - offset] = ec.getGPUPointerAddress(inputs.get(i));

//		if(isSinglePrecision()) {
//			float[] scalars = prepInputScalarsFloat(scalarObjects);
//
//			// ToDo: handle float
//		   ret = execute_f(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
//					in_ptrs, side_ptrs, out_ptr, scalars, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), out_obj.getNumColumns(),0);
//
//		}
//		else {
			double[] scalars = prepInputScalars(scalarObjects);

			long out_cols = out_obj == null ? 1 : out_obj.getNumColumns();
			long out_rows = out_obj == null ? 1 : out_obj.getNumRows();

			long out_len = out_rows * out_cols;

			ret = execute_d(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
					in_ptrs, offset, side_ptrs, out_ptr, scalars, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 
					out_len,0, inputs, out_obj);
//		}
		return ret;
	}

	@Override
	public String getSpoofType() {
		String[] tmp = getClass().getName().split("\\.");
		return  tmp[tmp.length-1] + "_" + getSpoofTemplateType() + "_" + name.split("\\.")[1];
	}

	private native float execute_f(long ctx, String name, long[] in_ptr, long[] side_ptr,
								   long out_ptr, float[] scalars, long m, long n, long out_len, long grix);

	private native double execute_d(long ctx, String name, long[] in_ptr, int offset, long[] side_ptr,
									long[] out_ptr, double[] scalars, long m, long n, long out_len, long grix,
									ArrayList<MatrixObject> inputs, MatrixObject output);
}
