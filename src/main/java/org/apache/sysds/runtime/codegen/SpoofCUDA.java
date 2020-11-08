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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class SpoofCUDA extends SpoofOperator {
	private static final long serialVersionUID = -2161276866245388359L;
	
	private final CNodeTpl cnt;
	public final String name;

	public SpoofCUDA(CNodeTpl cnode) {
		name = "codegen." + cnode.getVarname();
		cnt = cnode;
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
							   ExecutionContext ec) {
		double ret = 0;
		long out_ptr = 0;

		if(out_obj != null)
			out_ptr = ec.getGPUPointerAddress(out_obj);

		int offset = 1;
		if(cnt instanceof CNodeOuterProduct)
			offset = 2;

		// only dense input preparation for now
		long[] in_ptrs = new long[offset];
		for(int i = 0; i < offset; ++i)
			in_ptrs[i] = ec.getGPUPointerAddress(inputs.get(i));

		long[] side_ptrs = new long[inputs.size() - offset];
		for(int i = offset; i < inputs.size(); ++i)
			side_ptrs[i - offset] = ec.getGPUPointerAddress(inputs.get(i));

		if(isSinglePrecision()) {
			float[] scalars = prepInputScalarsFloat(scalarObjects);

			// ToDo: handle float
		   ret = execute_f(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
					in_ptrs, side_ptrs, out_ptr, scalars, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);

		}
		else {
			double[] scalars = prepInputScalars(scalarObjects);

			ret = execute_d(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
					in_ptrs, side_ptrs, out_ptr, scalars, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);
		}
		return ret;
	}

	@Override
	public String getSpoofType() {
		String tmp[] = getClass().getName().split("\\.");
			return  tmp[tmp.length-1] + "_" + getSpoofTemplateType() + "_" + name.split("\\.")[1];
	}

	private native float execute_f(long ctx, String name, long[] in_ptr, long[] side_ptr,
								   long out_ptr, float[] scalars, long m, long n, long grix);

	private native double execute_d(long ctx, String name, long[] in_ptr, long[] side_ptr,
									long out_ptr, double[] scalars, long m, long n, long grix);
}
