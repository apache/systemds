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
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;

import java.util.ArrayList;

public class SpoofCUDACellwise extends SpoofCellwise implements SpoofCUDAOperator {
	private static final long serialVersionUID = -5255791443086948200L;
	private static final Log LOG = LogFactory.getLog(SpoofCUDACellwise.class.getName());
	private final int ID;
	private final PrecisionProxy call;
	private Pointer ptr;
	private final SpoofCellwise fallback_java_op;
	
	public SpoofCUDACellwise(CellType type, boolean sparseSafe, boolean containsSeq, AggOp aggOp, int id,
			PrecisionProxy ep, SpoofCellwise fallback) {
		super(type, sparseSafe, containsSeq, aggOp);
		ID = id;
		call = ep;
		ptr = null;
		fallback_java_op = fallback;
	}
	
	@Override
	public ScalarObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects) {
		double[] result = new double[1];
		// ToDo: this is a temporary "solution" before perf opt
		int NT=256;
		long N = inputs.get(0).getNumRows() * inputs.get(0).getNumColumns();
		long num_blocks = ((N + NT * 2 - 1) / (NT * 2));
		Pointer ptr = ec.getGPUContext(0).allocate(getName(), LibMatrixCUDA.sizeOfDataType * num_blocks);
		long[] out = {1,1,1, 0, 0, GPUObject.getPointerAddress(ptr)};
		int offset = 1;
		if(call.exec(ec, this, ID, prepareInputPointers(ec, inputs, offset), 
			prepareSideInputPointers(ec, inputs, offset, false), out, scalarObjects, 0 ) != 0) {
			LOG.error("SpoofCUDA " + getSpoofType() + " operator failed to execute. Trying Java fallback.\n");
			// ToDo: java fallback
		}
		LibMatrixCUDA.cudaSupportFunctions.deviceToHost(ec.getGPUContext(0), ptr, result, getName(), false);
		
		return new DoubleObject(result[0]);
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
	public MatrixObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects,
			String outputName) {
		
		long out_rows = ec.getMatrixObject(outputName).getNumRows();
		long out_cols = ec.getMatrixObject(outputName).getNumColumns();
		MatrixObject a = inputs.get(0);
		GPUContext gctx = ec.getGPUContext(0);
		int m = (int) a.getNumRows();
		int n = (int) a.getNumColumns();
		double[] scalars = prepInputScalars(scalarObjects);
		if(_type == CellType.COL_AGG)
			out_rows = 1;
		else if(_type == SpoofCellwise.CellType.ROW_AGG)
			out_cols = 1;
		
		boolean sparseSafe = isSparseSafe() || ((inputs.size() < 2) && 
				genexec( 0, new SideInput[0], scalars, m, n, 0, 0 ) == 0);
		
//		ec.setMetaData(outputName, out_rows, out_cols);
		GPUObject g = a.getGPUObject(gctx);
		boolean sparseOut = _type == CellType.NO_AGG && sparseSafe && g.isSparse();
		
		long nnz = g.getNnz("spoofCUDA" + getSpoofType(), false);
		if(sparseOut)
			LOG.warn("sparse out");
		MatrixObject out_obj = sparseOut ?
				(ec.getSparseMatrixOutputForGPUInstruction(outputName, out_rows, out_cols, (isSparseSafe() && nnz > 0) ?
						nnz : out_rows * out_cols).getKey()) :
				(ec.getDenseMatrixOutputForGPUInstruction(outputName, out_rows, out_cols).getKey());
		
		int offset = 1;
		if(!inputIsEmpty(a.getGPUObject(gctx)) || !sparseSafe) {
			if(call.exec(ec, this, ID, prepareInputPointers(ec, inputs, offset), prepareSideInputPointers(ec, inputs, offset, false),
				prepareOutputPointers(ec, out_obj, sparseOut), scalarObjects, 0) != 0) {
				LOG.error("SpoofCUDA " + getSpoofType() + " operator failed to execute. Trying Java fallback.(ToDo)\n");
				// ToDo: java fallback
			}
		}
		return out_obj;
	}
	
	private static boolean inputIsEmpty(GPUObject g) {
		return g.getDensePointer() == null && g.getSparseMatrixCudaPointer() == null;
	}
	
	// used to determine sparse safety
	@Override 
	protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long gix, int rix, int cix) {
		return fallback_java_op.genexec(a, b, scalars, m, n, 0, 0, 0);
	}
	
	public int execute_sp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars) {
		return execute_f(ctx, meta, in, sides, out, scalars);	
	}
	
	public int execute_dp(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars) {
		return execute_d(ctx, meta, in, sides, out, scalars);
	}
	
	public static native int execute_f(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
	public static native int execute_d(long ctx, long[] meta, long[] in, long[] sides, long[] out, long scalars);
}
