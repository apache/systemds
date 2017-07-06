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

package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.lops.runtime.RunMRJobs;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.GPUInstructionParser;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;

public abstract class GPUInstruction extends Instruction
{
	public enum GPUINSTRUCTION_TYPE {
		AggregateUnary,
		AggregateBinary,
		RelationalBinary,
		Convolution,
		MMTSJ,
		Reorg,
		ArithmeticBinary,
		BuiltinUnary,
		BuiltinBinary,
		Builtin
	};

	// Memory/conversions
	public final static String MISC_TIMER_HOST_TO_DEVICE =          "H2D";	// time spent in bringing data to gpu (from host)
	public final static String MISC_TIMER_DEVICE_TO_HOST =          "D2H"; 	// time spent in bringing data from gpu (to host)
	public final static String MISC_TIMER_DEVICE_TO_DEVICE =        "D2D"; 	// time spent in copying data from one region on the device to another
	public final static String MISC_TIMER_SPARSE_TO_DENSE =         "s2d";	// time spent in converting data from sparse to dense
	public final static String MISC_TIMER_DENSE_TO_SPARSE =         "d2s";	// time spent in converting data from dense to sparse
	public final static String MISC_TIMER_ROW_TO_COLUMN_MAJOR =     "r2c";	// time spent in converting data from row major to column major
	public final static String MISC_TIMER_COLUMN_TO_ROW_MAJOR =     "c2r";	// time spent in converting data from column major to row major
	public final static String MISC_TIMER_OBJECT_CLONE =            "clone";// time spent in cloning (deep copying) a GPUObject instance

	public final static String MISC_TIMER_CUDA_FREE =               "f";		// time spent in calling cudaFree
	public final static String MISC_TIMER_ALLOCATE =                "a";		// time spent to allocate memory on gpu
	public final static String MISC_TIMER_ALLOCATE_DENSE_OUTPUT =   "ad";		// time spent to allocate dense output (recorded differently than MISC_TIMER_ALLOCATE)
	public final static String MISC_TIMER_ALLOCATE_SPARSE_OUTPUT =  "as";		// time spent to allocate sparse output (recorded differently than MISC_TIMER_ALLOCATE)
	public final static String MISC_TIMER_SET_ZERO =                "az";		// time spent to allocate
	public final static String MISC_TIMER_REUSE =                   "r";		// time spent in reusing already allocated memory on GPU (mainly for the count)

	// Matmult instructions
	public final static String MISC_TIMER_SPARSE_ALLOCATE_LIB = 						"Msao";		// time spend in allocating for sparse matrix output
	public final static String MISC_TIMER_DENSE_DOT_LIB = 									"Mddot";	// time spent in dot product of 2 dense vectors
	public final static String MISC_TIMER_DENSE_VECTOR_DENSE_MATRIX_LIB = 	"Mdvdm";	// time spent in matrix mult of dense vector and dense matrix
	public final static String MISC_TIMER_DENSE_MATRIX_DENSE_VECTOR_LIB = 	"Mdmdv";	// time spent in matrix mult of dense matrix and dense vector
	public final static String MISC_TIMER_DENSE_MATRIX_DENSE_MATRIX_LIB = 	"Mdmdm";	// time spent in matrix mult of dense matrices
	public final static String MISC_TIMER_SPARSE_MATRIX_DENSE_VECTOR_LIB = 	"Msmdv";	// time spent in matrix mult of sparse matrix and dense vector
	public final static String MISC_TIMER_SPARSE_MATRIX_SPARSE_MATRIX_LIB = "Msmsm";  // time spent in matrix mult of sparse matrices
	public final static String MISC_TIMER_SYRK_LIB = 												"Msyrk"; 	// time spent in symmetric rank-k update

	// Other BLAS instructions
	public final static String MISC_TIMER_DAXPY_LIB =   "daxpy";    // time spent in daxpy
	public final static String MISC_TIMER_QR_BUFFER =   "qr_buffer";// time spent in calculating buffer needed to perform QR
	public final static String MISC_TIMER_QR =          "qr";       // time spent in doing QR
	public final static String MISC_TIMER_ORMQR =       "ormqr";    // time spent in ormqr
	public final static String MISC_TIMER_TRSM =        "trsm";     // time spent in cublas Dtrsm

	// Transpose
	public final static String MISC_TIMER_SPARSE_DGEAM_LIB =    "sdgeaml";  // time spent in sparse transpose (and other ops of type a*op(A) + b*op(B))
	public final static String MISC_TIMER_DENSE_DGEAM_LIB =     "ddgeaml";  // time spent in dense transpose (and other ops of type a*op(A) + b*op(B))
	public final static String MISC_TIMER_TRANSPOSE_LIB =       "dtl";      // time spent on dense transpose, this includes allocation of output

	// Custom kernels
	public final static String MISC_TIMER_MATRIX_MATRIX_CELLWISE_OP_KERNEL = "mmck";   // time spent in matrix-matrix cellwise operations
	public final static String MISC_TIMER_COMPARE_AND_SET_KERNEL =           "cask";   // time spent in compareAndSet kernel
	public final static String MISC_TIMER_EXP_KERNEL =                       "expk";   // time spent in the exp kernel
	public final static String MISC_TIMER_SQRT_KERNEL =                      "sqrtk";   // time spent in the sqrt kernel
	public final static String MISC_TIMER_ROUND_KERNEL =                     "roundk";   // time spent in the round kernel
	public final static String MISC_TIMER_ABS_KERNEL =                       "absk";   // time spent in the abs kernel
	public final static String MISC_TIMER_LOG_KERNEL =                       "logk";   // time spent in the log kernel
	public final static String MISC_TIMER_FLOOR_KERNEL =                     "floork";  // time spent in the floor kernel
	public final static String MISC_TIMER_CEIL_KERNEL =                      "ceilk";   // time spent in the ceil kernel
	public final static String MISC_TIMER_SIN_KERNEL =                       "sink";   // time spent in the sin kernel
	public final static String MISC_TIMER_COS_KERNEL =                       "cosk";   // time spent in the cos kernel
	public final static String MISC_TIMER_TAN_KERNEL =                       "tank";   // time spent in the tan kernel
	public final static String MISC_TIMER_ASIN_KERNEL =                      "asink";   // time spent in the asin kernel
	public final static String MISC_TIMER_ACOS_KERNEL =                      "acosk";   // time spent in the acos kernel
	public final static String MISC_TIMER_ATAN_KERNEL =                      "atank";   // time spent in the atan kernel
	public final static String MISC_TIMER_SIGN_KERNEL =                      "signk";   // time spent in the sign kernel

	public final static String MISC_TIMER_DAXPY_MV_KERNEL =                  "daxpymv";// time spent in the daxpy_matrix_vector kernel
	public final static String MISC_TIMER_UPPER_TO_LOWER_TRIANGLE_KERNEL =   "u2lk";   // time spent in the copy_u2l_dense kernel
	public final static String MISC_TIMER_FILL_KERNEL =                      "fillk";  // time spent in the "fill" kernel
	public final static String MISC_TIMER_MATRIX_SCALAR_OP_KERNEL =          "msk";    // time spent in the matrix scalar kernel
	public final static String MISC_TIMER_REDUCE_ALL_KERNEL =                "rallk";  // time spent in reduce all kernel
	public final static String MISC_TIMER_REDUCE_ROW_KERNEL =                "rrowk";  // time spent in reduce row kernel
	public final static String MISC_TIMER_REDUCE_COL_KERNEL =                "rcolk";  // time spent in reduce column kernel

	// Deep learning operators
	public final static String MISC_TIMER_ACTIVATION_FORWARD_LIB =         "nnaf";  // time spent in cudnnActivationForward
	public final static String MISC_TIMER_CONVOLUTION_FORWARD_LIB =        "nncf";  // time spent in cudnnConvolutionForward
	public final static String MISC_TIMER_CONVOLUTION_BACKWARD_FILTER_LIB ="nncbf"; // time spent in cudnnConvolutionBackwardFilter
	public final static String MISC_TIMER_CONVOLUTION_BACKWARD_DATA_LIB =  "nncbd"; // time spent in cudnnConvolutionBackwardData
	public final static String MISC_TIMER_MAXPOOLING_FORWARD_LIB =         "nnmf";  // time spent in cudnnPoolingForward
	public final static String MISC_TIMER_MAXPOOLING_BACKWARD_LIB =        "nnmb";  // time spent in cudnnPoolingBackward
	public final static String MISC_TIMER_BIAS_ADD_LIB =                   "nnba";  // time spent in bias_add cuda kernel
	public final static String MISC_TIMER_RELU_BACKWARD_KERNEL=            "nnrbk"; // time spent in relu_backward cuda kernel
	public final static String MISC_TIMER_RELU_KERNEL =                    "nnrk";  // time spent in the relu kernel
	public final static String MISC_TIMER_CUDNN_INIT =                     "nni";   // time spent in initializations for cudnn call
	public final static String MISC_TIMER_CUDNN_CLEANUP =                  "nnc";   // time spent in cleanup for cudnn call


	protected GPUINSTRUCTION_TYPE _gputype;
	protected Operator _optr;

	protected boolean _requiresLabelUpdate = false;

	public GPUInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.GPU;
		instString = istr;

		//prepare opcode and update requirement for repeated usage
		instOpcode = opcode;
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}

	public GPUInstruction(Operator op, String opcode, String istr) {
		this(opcode, istr);
		_optr = op;
	}

	public GPUINSTRUCTION_TYPE getGPUInstructionType() {
		return _gputype;
	}

	@Override
	public boolean requiresLabelUpdate() {
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException
	{
		//default preprocess behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);

		//instruction patching
		if( tmp.requiresLabelUpdate() ) { //update labels only if required
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = RunMRJobs.updateLabels(tmp.toString(), ec.getVariables());
			tmp = GPUInstructionParser.parseSingleInstruction(updInst);
		}

		return tmp;
	}

	@Override
	public abstract void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException;

	@Override
	public void postprocessInstruction(ExecutionContext ec)
					throws DMLRuntimeException
	{
		//JCuda.cudaDeviceSynchronize();
	}

	/**
	 * Helper method to get the input block (allocated on the GPU)
	 * Also records performance information into {@link Statistics}
	 * @param ec		active {@link ExecutionContext}
	 * @param name	name of input matrix (that the {@link ExecutionContext} is aware of)
	 * @return	the matrix object
	 * @throws DMLRuntimeException if an error occurs
	 */
	protected MatrixObject getMatrixInputForGPUInstruction(ExecutionContext ec, String name) throws DMLRuntimeException {
		long t0 = System.nanoTime();
		Pair<MatrixObject, Boolean> mb = ec.getMatrixInputForGPUInstruction(name);
		if (mb.getValue()) GPUStatistics.maintainCPMiscTimes(getExtendedOpcode(), GPUInstruction.MISC_TIMER_HOST_TO_DEVICE, System.nanoTime() - t0);
		return mb.getKey();
	}

	/**
	 * Helper method to get the output block (allocated on the GPU)
	 * Also records performance information into {@link Statistics}
	 * @param ec		active {@link ExecutionContext}
	 * @param name	name of input matrix (that the {@link ExecutionContext} is aware of)
	 * @return	the matrix object
	 * @throws DMLRuntimeException	if an error occurs
	 */
	protected MatrixObject getDenseMatrixOutputForGPUInstruction(ExecutionContext ec, String name) throws DMLRuntimeException {
		long t0 = System.nanoTime();
		Pair<MatrixObject, Boolean> mb = ec.getDenseMatrixOutputForGPUInstruction(name);
		if (mb.getValue()) GPUStatistics.maintainCPMiscTimes(getExtendedOpcode(), GPUInstruction.MISC_TIMER_ALLOCATE_DENSE_OUTPUT, System.nanoTime() - t0);
		return mb.getKey();
	}
}
