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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;

import java.util.ArrayList;

public class SpoofCUDARowwise extends SpoofRowwise implements SpoofCUDAOperator {
	private static final long serialVersionUID = 3080001135814944399L;
	private static final Log LOG = LogFactory.getLog(SpoofCUDARowwise.class.getName());
	private final int ID;
	private final PrecisionProxy call;
	private Pointer ptr;
	
	public SpoofCUDARowwise(RowType type,  long constDim2, boolean tB1, int reqVectMem, int id,
		PrecisionProxy ep) {
		super(type, constDim2, tB1, reqVectMem);
		ID = id;
		call = ep;
		ptr = null;
	}
	
	@Override public String getName() {
		return getSpoofType();
	}
	
	@Override public void setScalarPtr(Pointer _ptr) {
		ptr = _ptr;
	}
	
	@Override public Pointer getScalarPtr() {
		return ptr;
	}
	
	@Override public void releaseScalarGPUMemory(ExecutionContext ec) {
		if(ptr != null) {
			ec.getGPUContext(0).cudaFreeHelper(getSpoofType(), ptr, DMLScript.EAGER_CUDA_FREE);
			ptr = null;
		}
	}
	
	@Override
	public ScalarObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects) {
		double[] result = new double[1];
		Pointer ptr = ec.getGPUContext(0).allocate(getName(), LibMatrixCUDA.sizeOfDataType);
		long[] out = {1,1,1, 0, 0, GPUObject.getPointerAddress(ptr)};
		int offset = 1;
		if(call.exec(ec, this, ID, prepareInputPointers(ec, inputs, offset), prepareSideInputPointers(ec, inputs, offset, _tB1),
				out, scalarObjects, 0) != 0) {
			LOG.error("SpoofCUDA " + getSpoofType() + " operator failed to execute. Trying Java fallback.\n");
			// ToDo: java fallback
		}
		LibMatrixCUDA.cudaSupportFunctions.deviceToHost(ec.getGPUContext(0), ptr, result, getName(), false);
		return new DoubleObject(result[0]);
	}
	
	@Override
	public MatrixObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects, String outputName) {
		
		int m = (int) inputs.get(0).getNumRows();
		int n = (int) inputs.get(0).getNumColumns();
		final int n2 = _type.isConstDim2(_constDim2) ? (int)_constDim2 : _type.isRowTypeB1() ||
				hasMatrixObjectSideInput(inputs) ? getMinColsMatrixObjectSideInputs(inputs) : -1;
		OutputDimensions out_dims = new OutputDimensions(m, n, n2);
		ec.setMetaData(outputName, out_dims.rows, out_dims.cols);
		MatrixObject out_obj = ec.getDenseMatrixOutputForGPUInstruction(outputName, out_dims.rows, out_dims.cols).getKey();
		
		int offset = 1;
		if(call.exec(ec,this, ID, prepareInputPointers(ec, inputs, offset), prepareSideInputPointers(ec, inputs, 
				offset, _tB1), prepareOutputPointers(ec, out_obj, false), scalarObjects, 0) != 0) {
			LOG.error("SpoofCUDA " + getSpoofType() + " operator failed to execute. Trying Java fallback.\n");
			// ToDo: java fallback
		}
		return out_obj;
	}
	
	// unused
	@Override protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len,
		long grix, int rix) { }

	// unused
	@Override protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c,
		int ci, int alen, int n, long grix, int rix) { }
	
	public int execute_sp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars) {
		return execute_f(ctx, meta, in, sides, out, scalars);
	}
	
	public int execute_dp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars) {
		return execute_d(ctx, meta, in, sides, out, scalars);
	}
	
	public static native int execute_f(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
	public static native int execute_d(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
}
