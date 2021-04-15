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
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;

import java.util.ArrayList;

public class SpoofCUDACellwise extends SpoofCellwise implements SpoofCUDAOperator {
	private static final long serialVersionUID = -5255791443086948200L;
	private static final Log LOG = LogFactory.getLog(SpoofCUDACellwise.class.getName());
	private final int ID;
	private final PrecisionProxy call;
	private final SpoofCellwise fallback_java_op;
	private final long ctx;
	
	public SpoofCUDACellwise(CellType type, boolean sparseSafe, boolean containsSeq, AggOp aggOp, int id,
		PrecisionProxy ep, SpoofCellwise fallback)
	{
		super(type, sparseSafe, containsSeq, aggOp);
		ID = id;
		call = ep;
		fallback_java_op = fallback;
		ctx = SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA);
	}
	
	@Override
	public ScalarObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects)
	{
		double[] result = new double[1];
		Pointer[] ptr = new Pointer[1];
		packDataForTransfer(ec, inputs, scalarObjects, null, 1, ID, 0,false, ptr);
		if(NotEmpty(inputs.get(0).getGPUObject(ec.getGPUContext(0)))) {
			if(call.exec(this) != 0)
				LOG.error("SpoofCUDA " + getSpoofType() + " operator " + ID + " failed to execute!\n");
		}
		LibMatrixCUDA.cudaSupportFunctions.deviceToHost(ec.getGPUContext(0), ptr[0], result, getName(), false);
		ec.getGPUContext(0).cudaFreeHelper(getSpoofType(), ptr[0], DMLScript.EAGER_CUDA_FREE);
		return new DoubleObject(result[0]);
	}
	
	@Override public String getName() {
		return getSpoofType();
	}

	@Override
	public MatrixObject execute(ExecutionContext ec, ArrayList<MatrixObject> inputs,
		ArrayList<ScalarObject> scalarObjects, String outputName)
	{
		long out_rows = ec.getMatrixObject(outputName).getNumRows();
		long out_cols = ec.getMatrixObject(outputName).getNumColumns();

		if(_type == CellType.COL_AGG)
			out_rows = 1;
		else if(_type == SpoofCellwise.CellType.ROW_AGG)
			out_cols = 1;

		double[] scalars = prepInputScalars(scalarObjects);
		boolean sparseSafe = isSparseSafe() || ((inputs.size() < 2) && 
				genexec( 0, new SideInput[0], scalars, (int) inputs.get(0).getNumRows(),
					(int) inputs.get(0).getNumColumns(), 0, 0 ) == 0);

		GPUObject in_obj = inputs.get(0).getGPUObject(ec.getGPUContext(0));
		boolean sparseOut = _type == CellType.NO_AGG && sparseSafe && in_obj.isSparse();
		long nnz = in_obj.getNnz("spoofCUDA" + getSpoofType(), false);
		MatrixObject out_obj = sparseOut ?
				(ec.getSparseMatrixOutputForGPUInstruction(outputName, out_rows, out_cols, (isSparseSafe() && nnz > 0) ?
						nnz : out_rows * out_cols).getKey()) :
				(ec.getDenseMatrixOutputForGPUInstruction(outputName, out_rows, out_cols).getKey());

		packDataForTransfer(ec, inputs, scalarObjects, out_obj, 1, ID, 0,false, null);
		if(NotEmpty(in_obj) || !sparseSafe) {
			if(call.exec(this) != 0)
				LOG.error("SpoofCUDA " + getSpoofType() + " operator " + ID + " failed to execute!\n");
		}
		return out_obj;
	}
	
	private static boolean NotEmpty(GPUObject g) {
		// ToDo: check if that check is sufficient
		return g.getDensePointer() != null || g.getSparseMatrixCudaPointer() != null;
	}
	
	// used to determine sparse safety
	@Override 
	protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long gix, int rix, int cix) {
		return fallback_java_op.genexec(a, b, scalars, m, n, 0, 0, 0);
	}

	public int execute_dp(long ctx) { return execute_d(ctx); }
	public int execute_sp(long ctx) { return execute_d(ctx); }
	public long getContext() { return ctx; }

	public static native int execute_d(long ctx);
	public static native int execute_s(long ctx);
}
