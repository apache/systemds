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

package org.apache.sysds.hops;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.DnnTransform;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.matrix.data.DnnParameters;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class DnnOp extends MultiThreadedHop {
	private static final Log LOG =  LogFactory.getLog(DnnOp.class.getName());

	// -------------------------------------------------------------------------
	// This flag allows us to compile plans with less unknowns and also serves as future tensorblock integration.
	// By default, these flags are turned on.
	
	// When this flag is turned on, we attempt to check the parent convolution hop for unknown dimensions.
	// For example: in case of conv -> maxpool, the input channel/height/width of maxpool will match output channel/height/width of conv.
	private static final boolean INFER_TENSOR_SHAPE_FROM_PARENT_CONV_OP = true;
	// This guards us from cases where the user provides incorrect C,H,W parameters.
	private static final boolean THROW_ERROR_IF_INFERRED_SHAPE_MISMATCH = true;
	// -------------------------------------------------------------------------
	
	// Specifies the type of this hop
	private OpOpDnn op;

	private DnnOp() {
		//default constructor for clone
	}

	/**
	 * Create a hop from the builtin expression
	 * 
	 * @param l name of the hop
	 * @param dt datatype (only supports matrix datatype)
	 * @param vt valuetype  (only supports matrix valuetype) 
	 * @param o type of this hop
	 * @param inp input hops
	 */
	public DnnOp(String l, DataType dt, ValueType vt, OpOpDnn o, ArrayList<Hop> inp) 
	{
		super(l, dt, vt);
		op = o;
		
		for( int i=0; i<inp.size(); i++ ) {
			Hop in = inp.get(i);
			getInput().add(i, in);
			in.getParent().add(this);
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public OpOpDnn getOp() {
		return op;
	}
	
	@Override
	public String getOpString() {
		return op.toString();
	}

	private static boolean isEligibleForSpark() {
		return false;
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		return true;
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return true;
	}
	
	@Override
	public Lop constructLops()
	{
		//return already created lops
		if( getLops() != null )
			return getLops();
		
		ExecType et = optFindExecType();
		
		switch( op ) {
			case MAX_POOL:
			case MAX_POOL_BACKWARD:
			case AVG_POOL:
			case AVG_POOL_BACKWARD:
			case CONV2D:
			case CONV2D_BACKWARD_DATA:
			case CONV2D_BACKWARD_FILTER:
			case BIASADD:
			case BIASMULT: {
				if(et == ExecType.CP || et == ExecType.GPU) {
					setLops(constructDnnLops(et, getInput()));
					break;
				}
				throw new HopsException("Unimplemented DnnOp for execution type: " + et.name());
			}
			case BATCH_NORM2D_TEST:
			case CHANNEL_SUMS:
			case UPDATE_NESTEROV_X: {
				if(et == ExecType.GPU) {
					setLops(constructDnnLops(et, getInput()));
					break;
				}
				throw new HopsException("Unimplemented DnnOp for execution type: " + et.name());
			}
			default:
				throw new HopsException("Unsupported lops construction for operation type '"+op+"'.");
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}
	
	public void setOp(OpOpDnn op) {
		this.op = op;
	}
	
	private int getNumExpectedInputs() {
		switch(op) {
			case MAX_POOL_BACKWARD:
			case AVG_POOL_BACKWARD:
			case CONV2D:
			case CONV2D_BACKWARD_FILTER:
			case CONV2D_BACKWARD_DATA:
				return 14;
			case BIASADD:
			case BIASMULT:
				return 2;
			case BATCH_NORM2D_TEST:
				return 6;
			case CHANNEL_SUMS:
				return 3;
			case UPDATE_NESTEROV_X:
				return 4;
			default:
				return 13;
		}
	}
	
	/**
	 * Returns parent matrix X or null
	 * @param input input hop
	 * @return either null or X if input is max(X,0) or max(0,X)
	 */
	private static Hop isInputReLU(Hop input) {
		if(HopRewriteUtils.isBinary(input, OpOp2.MAX)) {
			if(HopRewriteUtils.isLiteralOfValue(input.getInput().get(0), 0)) {
				return input.getInput().get(1);
			}
			else if(HopRewriteUtils.isLiteralOfValue(input.getInput().get(1), 0)) {
				return input.getInput().get(0);
			}
			else
				return null; 
		}
		else
			return null;
	}
	
	private static boolean isInputConv2d(Hop input) {
		return HopRewriteUtils.isDnn(input, OpOpDnn.CONV2D);
	}
	
	/**
	 * Compares the input parameters for max_pool/max_pool_backward operations
	 * 
	 * @return true if the following parameters match: stride=[stride, stride], padding=[pad, pad], input_shape=[numImg, numChannels, imgSize, imgSize], pool_size=[poolSize1, poolSize2]
	 */
	private static boolean isPoolingParametersEqualAndKnown(DnnParameters param1, DnnParameters param2) {
		return isEqualAndKnown(param1.stride_h, param2.stride_h) && isEqualAndKnown(param1.stride_w, param2.stride_w) && 
			isEqualAndKnown(param1.pad_h, param2.pad_h) && isEqualAndKnown(param1.pad_w, param2.pad_w) &&
			isEqualAndKnown(param1.R, param2.R) && isEqualAndKnown(param1.S, param2.S) &&
			isEqualAndKnown(param1.N, param2.N) && isEqualAndKnown(param1.C, param2.C) &&
			isEqualAndKnown(param1.H, param2.H) && isEqualAndKnown(param1.W, param2.W);
	}
	
	public boolean isStride1Pad0() {
		DnnParameters tmp = parseInput();
		return tmp.stride_h == 1 && tmp.stride_w == 1
			&& tmp.pad_h == 0 && tmp.pad_w == 0;
	}
	
	private static boolean isEqualAndKnown(int val1, int val2) {
		return val1 >= 0 && val2 >= 0 && val1 == val2;
	}
	
	/**
	 * Returns the output lop of max_pool/avg_pool operation with same parameters as this hop.
	 * If corresponding output lop is not found or if this is not a max_pool_backward operation, this function returns null
	 * 
	 * @return output lop of max_pool/avg_pool operation with same parameters as this hop
	 */
	private Lop getMaxPoolOutputLop() {
		if(op == OpOpDnn.MAX_POOL_BACKWARD || op == OpOpDnn.AVG_POOL_BACKWARD) {
			OpOpDnn opType = (op == OpOpDnn.MAX_POOL_BACKWARD) ? OpOpDnn.MAX_POOL : OpOpDnn.AVG_POOL;
			Hop inputImage = getInput().get(0);
			for(Hop tmpParent : inputImage.getParent()) {
				if(!(tmpParent instanceof DnnOp))
					continue;
				DnnOp parent = (DnnOp) tmpParent;
				if(parent.getOp() == opType && isPoolingParametersEqualAndKnown(parent._cachedParams, _cachedParams)) {
					return parent.constructLops();
				}
			}
		}
		return null;
	}
	
	public Lop constructDnnLops(ExecType et, List<Hop> inputs) {
		if(inputs.size() != getNumExpectedInputs()) 
			throw new HopsException("Incorrect number of inputs for " + op.name());
		
		//TODO move these custom rewrites to the general hop rewrites
		// ---------------------------------------------------------------
		// Deal with fused operators and contruct lhsInputLop/optionalRhsInputLop
		Lop lhsInputLop = null; Lop optionalRhsInputLop = null;
		List<Hop> inputsOfPotentiallyFusedOp = inputs;
		
		OpOpDnn lopOp = op;
		// RELU_MAX_POOLING and RELU_MAX_POOLING_BACKWARD is extremely useful for CP backend 
		// by reducing unnecessary sparse-to-dense-to-sparse conversion.
		// For other backends, this operators is not necessary as it reduces an additional relu operator.
		Hop parentReLU = isInputReLU(inputs.get(0));
		if(OptimizerUtils.ALLOW_OPERATOR_FUSION && et == ExecType.CP && op == OpOpDnn.MAX_POOL && parentReLU != null) {
			lhsInputLop = parentReLU.constructLops();
			lopOp = OpOpDnn.RELU_MAX_POOL;
		}
		else if(OptimizerUtils.ALLOW_OPERATOR_FUSION && et == ExecType.CP && op == OpOpDnn.MAX_POOL_BACKWARD && parentReLU != null) {
			lhsInputLop = parentReLU.constructLops();
			lopOp = OpOpDnn.RELU_MAX_POOL_BACKWARD;
		}
		else if(OptimizerUtils.ALLOW_OPERATOR_FUSION && op == OpOpDnn.BIASADD && isInputConv2d(inputs.get(0))) {
			lopOp = OpOpDnn.CONV2D_BIAS_ADD;
			
			// the first lop is image 
			lhsInputLop = inputs.get(0).getInput().get(0).constructLops();
			// the second lop is bias
			optionalRhsInputLop = inputs.get(1).constructLops();
			
			// Use the inputs from conv2d rather than bias_add
			inputsOfPotentiallyFusedOp = inputs.get(0).getInput();
		}
		else {
			lhsInputLop = inputs.get(0).constructLops();
		}
		// ---------------------------------------------------------------
		
		// ---------------------------------------------------------------
		// Compute intermediate memory budget that can be passed to GPU operators 
		// for better CuDNN operator selection at runtime
		double intermediateMemEstimate = computeIntermediateMemEstimate(-1, -1, -1 );
		if(et == ExecType.GPU && getDim1() >= 0 && getDim2() >= 0) {
			// This enables us to compile more efficient matrix-matrix CuDNN operation instead of 
			// row-by-row invocation of multiple vector-matrix CuDNN operations.
			// This is possible as the operations on GPU are single-threaded
			double optimisticIntermediateMemEstimate = GPUContextPool.initialGPUMemBudget() - getOutputMemEstimate() - inputs.get(0).getOutputMemEstimate();
			if(optionalRhsInputLop != null) {
				optimisticIntermediateMemEstimate -= inputs.get(1).getOutputMemEstimate();
			}
			intermediateMemEstimate = Math.max(intermediateMemEstimate, optimisticIntermediateMemEstimate);
		}
		// ---------------------------------------------------------------
		
		// Construct the lop
		Lop optionalMaxPoolOutput = (et == ExecType.GPU) ? getMaxPoolOutputLop() : null;
		Lop[] l2inputs = new Lop[inputsOfPotentiallyFusedOp.size()-1];
		for( int i=1; i < inputsOfPotentiallyFusedOp.size(); i++ )
			l2inputs[i-1] = inputsOfPotentiallyFusedOp.get(i).constructLops();
		DnnTransform convolutionLop = new DnnTransform(
			lhsInputLop, lopOp, getDataType(), getValueType(), et,
			OptimizerUtils.getConstrainedNumThreads(_maxNumThreads), intermediateMemEstimate);
		setOutputDimensions(convolutionLop);
		setLineNumbers(convolutionLop);
		
		// ---------------------------------------------------------------
		// Add input/output for parent lops of convolutionLop
		lhsInputLop.addOutput(convolutionLop);
		if(optionalRhsInputLop != null) {
			convolutionLop.addInput(optionalRhsInputLop);
			optionalRhsInputLop.addOutput(convolutionLop);
		}
		for( int i=0; i < l2inputs.length; i++ ) {
			convolutionLop.addInput(l2inputs[i]);
			l2inputs[i].addOutput(convolutionLop);
		}
		// Only valid for MAX_POOLING_BACKWARD on GPU
		if(optionalMaxPoolOutput != null) {
			convolutionLop.addInput(optionalMaxPoolOutput);
			optionalMaxPoolOutput.addOutput(convolutionLop);
		}
		convolutionLop.updateLopProperties();
		
		// TODO double check that optionalMaxPoolOutput adheres to proper
		// ID ordering of constructed lops (previously hidden by setLevel)
		
		// ---------------------------------------------------------------
		
		return convolutionLop;
	}

			
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		if(getOp() == OpOpDnn.BIASMULT) {
			// in non-gpu mode, the worst case size of bias multiply operation is same as that of input.
			if(DMLScript.USE_ACCELERATOR) 
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);
			else
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, getInput().get(0).getSparsity());
		}
		else {
			double sparsity = 1.0;
			return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
		}
	}
	
	// ---------------------------------------------------------------
	// Utility methods to guard the computation of memory estimates in presense of unknowns
	private static class IntermediateDimensions {
		int dim1; int dim2; double sp;
		public IntermediateDimensions(DnnOp h, String dim1Str, String dim2Str, double sp) {
			dim1 = (int) h.getDim(dim1Str);
			dim2 = (int) h.getDim(dim2Str);
			this.sp = sp;
		}
		public IntermediateDimensions(DnnOp h, String dim1Str, String dim2Str) {
			dim1 = (int) h.getDim(dim1Str);
			dim2 = (int) h.getDim(dim2Str);
			sp = 1;
		}
		public IntermediateDimensions(DnnOp h, int dim1, String dim2Str) {
			this.dim1 = dim1;
			dim2 = (int) h.getDim(dim2Str);
			sp = 1;
		}
		
		/**
		 * Add two computed memory estimates
		 * 
		 * @param val1 memory estimate 1
		 * @param val2 memory estimate 2
		 * @return sum of memory estimates
		 */
		static double guardedAdd(double val1, double val2) {
			if(val1 < 0 || val2 < 0) return OptimizerUtils.DEFAULT_SIZE;
			double ret = val1 + val2;
			if(ret >= OptimizerUtils.DEFAULT_SIZE) return OptimizerUtils.DEFAULT_SIZE;
			else return ret;
		}
		
		/**
		 * Compute memory estimates for given intermediate matrices 
		 * 
		 * @param intermediates list of intermediates
		 * @param numWorkers number of workers
		 * @return memory estimate
		 */
		public static double addEstimateSizes(ArrayList<IntermediateDimensions> intermediates, int numWorkers) {
			double memBudget = 0; 
			for(int i = 0; i < intermediates.size(); i++) {
				memBudget = guardedAdd(memBudget, OptimizerUtils.estimateSizeExactSparsity(
						intermediates.get(i).dim1, intermediates.get(i).dim2, intermediates.get(i).sp)*numWorkers);
			}
			return memBudget;
		}
		
		/**
		 * Compute max of two computed memory estimates
		 * @param val1 memory estimate 1
		 * @param val2 memory estimate 2
		 * @return max of memory estimates
		 */
		public static double guardedMax(double val1, double val2) {
			if(val1 < 0 || val2 < 0) return OptimizerUtils.DEFAULT_SIZE;
			double ret = Math.max(val1, val2);
			if(ret >= OptimizerUtils.DEFAULT_SIZE) return OptimizerUtils.DEFAULT_SIZE;
			else return ret;
		}
	}
	
	/**
	 * Helper utility to compute intermediate memory estimate
	 * 
	 * @param gpuIntermediates intermediates for GPU
	 * @param cpIntermediates intermediates for CP
	 * @return memory estimates
	 */
	private double computeIntermediateMemEstimateHelper(
			ArrayList<IntermediateDimensions> gpuIntermediates,
			ArrayList<IntermediateDimensions> cpIntermediates) {
		// Since CP operators use row-level parallelism by default
		int numWorkers = (int) Math.min(OptimizerUtils.getConstrainedNumThreads(_maxNumThreads), Math.max(getDim("N"), 1));
		if(DMLScript.USE_ACCELERATOR) {
			// Account for potential sparse-to-dense conversion
			double gpuMemBudget = IntermediateDimensions.addEstimateSizes(gpuIntermediates, 1);
			double cpMemoryBudget = IntermediateDimensions.addEstimateSizes(cpIntermediates, numWorkers);
			if(cpMemoryBudget > gpuMemBudget) {
				double oneThreadCPMemBudget = IntermediateDimensions.addEstimateSizes(cpIntermediates, 1);
				if(oneThreadCPMemBudget <= gpuMemBudget) {
					// Why limit CPU ? in-order to give more opportunity to compile GPU operators
					cpMemoryBudget = oneThreadCPMemBudget;
				}
			}
			// Finally, use the maximum of CP and GPU memory budget
			return IntermediateDimensions.guardedMax(cpMemoryBudget, gpuMemBudget);
		}
		else {
			// When -gpu flag is not provided, the memory estimates for CP are not affected.
			return IntermediateDimensions.addEstimateSizes(cpIntermediates, numWorkers);
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long ignoreDim1, long ignoreDim2, long ignoreNnz )
	{	
		ArrayList<IntermediateDimensions> gpuIntermediates = new ArrayList<>();
		ArrayList<IntermediateDimensions> cpIntermediates = new ArrayList<>();
		if(getOp() == OpOpDnn.CONV2D) {
			// Assumption: To compile a GPU conv2d operator, following should fit on the GPU:
			// 1. output in dense format (i.e. computeOutputMemEstimate) 
			// 2. input in any format
			// 3. atleast one input row in dense format
			// 4. filter in dense format
			
			// Account for potential sparse-to-dense conversion of atleast 1 input row and filter
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
			gpuIntermediates.add(new IntermediateDimensions(this, "K", "CRS"));
			
			// im2col operation preserves the worst-case sparsity of the input.
			cpIntermediates.add(new IntermediateDimensions(this, "CRS", "PQ", getInput().get(0).getSparsity()));
		}
		else if(getOp() == OpOpDnn.CONV2D_BACKWARD_DATA) {
			// Assumption: To compile a GPU conv2d_backward_data operator, following should fit on the GPU:
			// 1. output in dense format (i.e. computeOutputMemEstimate) 
			// 2. dout in any format
			// 3. atleast one dout row in dense format
			// 4. filter in dense format
			
			// Account for potential sparse-to-dense conversion of atleast 1 input row and filter
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "KPQ"));
			gpuIntermediates.add(new IntermediateDimensions(this, "K", "CRS"));
			
			// There are 2 intermediates: rotate180 and input to col2im for conv2d_backward_data
			// rotate180 preserves the "exact" sparsity of the dout matrix
			cpIntermediates.add(new IntermediateDimensions(this, "PQ", "K", getInput().get(1).getSparsity()));
			// Note: worst-case sparsity for the input of col2im (of size NPQ x CRS where N is determined by degree of parallelism)
			cpIntermediates.add(new IntermediateDimensions(this, "PQ", "CRS"));
		}
		else if(getOp() == OpOpDnn.CONV2D_BACKWARD_FILTER) {
			// Assumption: To compile a GPU conv2d_backward_filter operator, following should fit on the GPU:
			// 1. output in dense format (i.e. computeOutputMemEstimate) 
			// 2. dout in any format
			// 3. atleast one dout and input row in dense format
			
			// Account for potential sparse-to-dense conversion of atleast 1 input + dout row
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "KPQ"));
			
			// There are 2 intermediates: im2col and rotate180 for conv2d_backward_filter
			// rotate180 preserves the "exact" sparsity of the dout matrix
			cpIntermediates.add(new IntermediateDimensions(this, "PQ", "K", getInput().get(1).getSparsity()));
			// im2col operation preserves the worst-case sparsity of the input.
			cpIntermediates.add(new IntermediateDimensions(this, "CRS", "PQ", getInput().get(0).getSparsity()));
		}
		else if(getOp() == OpOpDnn.MAX_POOL || getOp() == OpOpDnn.AVG_POOL) {
			// Account for potential sparse-to-dense conversion of at least 1 input row
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
		}
		else if(getOp() == OpOpDnn.MAX_POOL_BACKWARD || getOp() == OpOpDnn.AVG_POOL_BACKWARD) {
			// Account for potential sparse-to-dense conversion of at least 1 input + dout row
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CPQ"));
		}
		
		if(gpuIntermediates.size() > 0 || cpIntermediates.size() > 0)
			return computeIntermediateMemEstimateHelper(gpuIntermediates, cpIntermediates);
		else
			return 0;
	}
	
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		// [numRows, numCols, NNZ] 
		DataCharacteristics ret = new MatrixCharacteristics();
		
		if(op == OpOpDnn.BIASADD || op == OpOpDnn.BIASMULT || op == OpOpDnn.BATCH_NORM2D_TEST ||
			op == OpOpDnn.UPDATE_NESTEROV_X) {
			// Same dimension as the first input
			DataCharacteristics[] mc = memo.getAllInputStats(getInput());
			ret = new MatrixCharacteristics(
				mc[0].rowsKnown() ? mc[0].getRows() : -1,
				mc[0].colsKnown() ? mc[0].getCols() : -1, -1, -1);
			return ret.dimsKnown() ? ret : null;
		}
		else if(op == OpOpDnn.CHANNEL_SUMS) {
			long numChannels = Hop.computeSizeInformation(getInput().get(1));
			return new MatrixCharacteristics(numChannels, 1, -1, -1);
		}
		
		//safe return (create entry only if at least dims known)
		refreshSizeInformation();
		ret = _dc;
		return ret.dimsKnown() ? ret : null;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType(boolean transitive) {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) {
			_etype = _etypeForced;
		}
		else {
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else {
				_etype = ExecType.SPARK;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		// TODO: Fix this after adding remaining spark instructions
		_etype = !isEligibleForSpark() && _etype == ExecType.SPARK ?  ExecType.CP : _etype;
		
		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		return _etype;
	}
	
	// Parameters recomputed in refreshSizeInformation and passed across many calls of getDim
	private DnnParameters _cachedParams = new DnnParameters(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, _maxNumThreads);
	
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4
	DnnParameters parseInput() {
		
		if(op == OpOpDnn.MAX_POOL_BACKWARD || op == OpOpDnn.AVG_POOL_BACKWARD 
				|| op == OpOpDnn.CONV2D 
				|| op == OpOpDnn.CONV2D_BACKWARD_FILTER
				|| op == OpOpDnn.CONV2D_BACKWARD_DATA) {
			_cachedParams.setIfUnknown(
					getInput().get(6),  // N
					getInput().get(7),  // C
					getInput().get(8),  // H
					getInput().get(9),  // W
					getInput().get(10), // K
					getInput().get(12), // R
					getInput().get(13), // S
					getInput().get(2),  // stride_h
					getInput().get(3),  // stride_w
					getInput().get(4),  // pad+h
					getInput().get(5), _maxNumThreads);
		}
		else {
			_cachedParams.setIfUnknown(
					getInput().get(5),
					getInput().get(6),
					getInput().get(7),
					getInput().get(8),
					getInput().get(9),
					getInput().get(11),
					getInput().get(12),
					getInput().get(1),
					getInput().get(2),
					getInput().get(3),
					getInput().get(4), _maxNumThreads);
		}
		
		if(INFER_TENSOR_SHAPE_FROM_PARENT_CONV_OP) {
			boolean isPool = (getOp() == OpOpDnn.MAX_POOL || getOp() == OpOpDnn.AVG_POOL);
			boolean isConv = getOp() == OpOpDnn.CONV2D;
			boolean unknownCHWPQ = _cachedParams.C < 0 || _cachedParams.H < 0 || _cachedParams.W < 0 || _cachedParams.P < 0 || _cachedParams.Q < 0;
			if((isPool || isConv) && unknownCHWPQ) {
				// Only infer input shape for convolution and maxpool
				inferCHWPQFromParentOp();
			}
		}
		
		if(_cachedParams.R < 0 && _cachedParams.H > 0) {
			// Unknown R, but known H and both are equal
			// This happens for one-dimensional conv2d where H=R and H can be inferred from the parent hop
			_cachedParams.R = _cachedParams.H;
		}
		
		// Compute P and Q if unknown. At script level, they are computed using following script:
		// P = as.integer(floor((H + 2*pad_h - R)/stride_h + 1))
		// Q = as.integer(floor((W + 2*pad_w - S)/stride_w + 1))
		if(_cachedParams.P < 0 && _cachedParams.H >= 0 && _cachedParams.R >= 0 && _cachedParams.stride_h >= 0 && _cachedParams.pad_h >= 0) {
			_cachedParams.P = (int) org.apache.sysds.runtime.util.DnnUtils.getP(_cachedParams.H, _cachedParams.R, _cachedParams.stride_h, _cachedParams.pad_h);
		}
		if(_cachedParams.Q < 0 && _cachedParams.W >= 0 && _cachedParams.S >= 0 && _cachedParams.stride_w >= 0 && _cachedParams.pad_w >= 0) {
			_cachedParams.Q = (int) org.apache.sysds.runtime.util.DnnUtils.getQ(_cachedParams.W, _cachedParams.S, _cachedParams.stride_w, _cachedParams.pad_w);
		}
		
		return _cachedParams;
	}
	
	/**
	 * Utility method to check if the given hop is a BIAS_ADD hop
	 * 
	 * @param hop the given hop
	 * @return true if the given hop is BIAS_ADD
	 */
	private static boolean isInputBiasAdd(Hop hop) {
		return HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD);
	}
	
	/**
	 * Utility method to check if the inferred shapes are equal to the given shape with a guard for unknown
	 * 
	 * @param dim1 inferred shape
	 * @param dim2 given shape
	 * @param paramType string denoting the parameter for pretty printing of the error message
	 */
	private static void throwExceptionIfNotEqual(int dim1, int dim2, String paramType) {
		if(dim1 >= 0 && dim2 >= 0 && dim1 != dim2) {
			throw new DMLRuntimeException("Inferred " + paramType + " from parent doesn't match with given " + paramType + ":" + dim1 + " != " + dim2);
		}
	}
	
	/**
	 * Gets the values for the parameters C, H, W, P, Q from parent hops
	 */
	private void inferCHWPQFromParentOp() {
		Hop tmp = getInput().get(0);
		// Skip bias_add and go to its parent
		tmp = isInputBiasAdd(tmp) ? tmp.getInput().get(0) : tmp;
		Hop parentReLU = isInputReLU(tmp);
		// Skip ReLU and go to its parent
		tmp =  (parentReLU != null) ? parentReLU : tmp;
		
		// Cast tmp as parent
		DnnOp parentOp = (tmp instanceof DnnOp) ? ((DnnOp) tmp) : null; 
		
		if(parentOp == null)
			return;
		else if(parentOp.getOp() == OpOpDnn.MAX_POOL || parentOp.getOp() == OpOpDnn.AVG_POOL) {
			DnnParameters parentParam = parentOp.parseInput();
			int prevC = _cachedParams.C; int prevH = _cachedParams.H; int prevW = _cachedParams.W;
			// [C, P, Q] from maxpool becomes [C, H, W] of next op
			_cachedParams.C = (_cachedParams.C < 0) ? parentParam.C : _cachedParams.C;
			_cachedParams.H = (_cachedParams.H < 0) ? parentParam.P : _cachedParams.H;
			_cachedParams.W = (_cachedParams.W < 0) ? parentParam.Q : _cachedParams.W;
			if(LOG.isDebugEnabled()) {
				LOG.debug("Inferring [C,H,W] from maxpool parent: [" + prevC + "," + prevH + "," + prevW + "]-> [" + _cachedParams.C + "," + _cachedParams.H + "," + _cachedParams.W + "]");
			}
			if(THROW_ERROR_IF_INFERRED_SHAPE_MISMATCH) {
				throwExceptionIfNotEqual(prevC, _cachedParams.C, "C");
				throwExceptionIfNotEqual(prevH, _cachedParams.H, "H");
				throwExceptionIfNotEqual(prevW, _cachedParams.W, "W");
			}
		}
		else if(parentOp.getOp() == OpOpDnn.CONV2D) {
			DnnParameters parentParam = parentOp.parseInput();
			int prevC = _cachedParams.C; int prevH = _cachedParams.H; int prevW = _cachedParams.W;
			// [K, P, Q] from convolution becomes [C, H, W] of next op
			_cachedParams.C = (_cachedParams.C < 0) ? parentParam.K : _cachedParams.C;
			_cachedParams.H = (_cachedParams.H < 0) ? parentParam.P : _cachedParams.H;
			_cachedParams.W = (_cachedParams.W < 0) ? parentParam.Q : _cachedParams.W;
			if(LOG.isDebugEnabled()) {
				LOG.debug("Inferring [C,H,W] from maxpool parent: [" + prevC + "," + prevH + "," + prevW + "]-> [" + _cachedParams.C + "," + _cachedParams.H + "," + _cachedParams.W + "]");
			}
			if(THROW_ERROR_IF_INFERRED_SHAPE_MISMATCH) {
				throwExceptionIfNotEqual(prevC, _cachedParams.C, "C");
				throwExceptionIfNotEqual(prevH, _cachedParams.H, "H");
				throwExceptionIfNotEqual(prevW, _cachedParams.W, "W");
			}
		}
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if(op == OpOpDnn.BIASADD || op == OpOpDnn.BIASMULT 
			|| op == OpOpDnn.BATCH_NORM2D_TEST || op == OpOpDnn.UPDATE_NESTEROV_X) {
			// Same dimension as the first input
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			setNnz(-1); // cannot infer stats
			return;
		}
		else if(op == OpOpDnn.CHANNEL_SUMS) {
			long numChannels = Hop.computeSizeInformation(getInput().get(1));
			setDim1(numChannels);
			setDim2(1);
			setNnz(-1); // cannot infer stats
			return;
		}
		
		// Reset the _cachedParams to avoid incorrect sizes
		_cachedParams = new DnnParameters(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, _maxNumThreads);
		
		switch(op) 
		{
			case MAX_POOL:
			case AVG_POOL: {
				setDim1(getDim("N"));
				setDim2(getDim("CPQ"));
				setNnz(-1); // cannot infer stats
				break;
			}
			case MAX_POOL_BACKWARD:
			case AVG_POOL_BACKWARD: {
				setDim1(getDim("N"));
				setDim2(getDim("CHW"));
				setNnz(-1);
				break;
			}
			case CONV2D: {
				setDim1(getDim("N"));
				setDim2(getDim("KPQ"));
				setNnz(-1); // cannot infer stats
				break;
			}
			case CONV2D_BACKWARD_DATA: {
				setDim1(getDim("N"));
				setDim2(getDim("CHW"));
				setNnz(-1); // cannot infer stats
				break;
			}
			case CONV2D_BACKWARD_FILTER: {
				setDim1(getDim("K"));
				setDim2(getDim("CRS"));
				setNnz(-1); // cannot infer stats
				break;
			}
			default:
				throw new RuntimeException("The sizes are not refreshed for " + op.name());
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException {
		DnnOp ret = new DnnOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		ret._maxNumThreads = _maxNumThreads;
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof DnnOp) )
			return false;
		
		DnnOp that2 = (DnnOp)that;
		
		boolean ret =  (op == that2.op)
				    && (getInput().size()==that.getInput().size())
				    && _maxNumThreads == that2._maxNumThreads;
		
		//compare all childs
		if( ret ) //sizes matched
			for( int i=0; i<_input.size(); i++ )
				ret &= getInput().get(i) == that2.getInput().get(i);
		
		return ret;
	}
	
	// ------------------------------------------------------------------------------------------------------
	// Utility methods to get the dimensions taking into account unknown dimensions
	
	/**
	 * Convenient method to get the dimensions required by ConvolutionOp.
	 * 
	 * @param dimString can be K, CRS, N, CHW, KPQ, PQ
	 * @return either -1 or value associated with the dimString
	 */
	private long getDim(String dimString) {
		if(op == OpOpDnn.BIASADD || op == OpOpDnn.BIASMULT
			|| op == OpOpDnn.BATCH_NORM2D_TEST || op == OpOpDnn.CHANNEL_SUMS ||
			op == OpOpDnn.UPDATE_NESTEROV_X) {
			throw new RuntimeException("getDim method should not be invoked for " + op.name());
		}
		try {
			parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		Hop filter = null; 	// shape: K x CRS 
		Hop input = null; 	// shape: N x CHW
		Hop dout = null;	// shape: N x KPQ
		Hop dout1 = null;	// shape: N x CPQ
		
		if(getOp() == OpOpDnn.CONV2D) {
			input  = getInput().get(0);
			filter = getInput().get(1);
		}
		else if(getOp() == OpOpDnn.CONV2D_BACKWARD_DATA) {
			filter = getInput().get(0);
			dout  = getInput().get(1);
		}
		else if(getOp() == OpOpDnn.CONV2D_BACKWARD_FILTER) {
			input = getInput().get(0);
			dout  = getInput().get(1);
		}
		else if(getOp() == OpOpDnn.MAX_POOL || getOp() == OpOpDnn.AVG_POOL) {
			input = getInput().get(0);
		}
		else if(getOp() == OpOpDnn.MAX_POOL_BACKWARD || getOp() == OpOpDnn.AVG_POOL_BACKWARD) {
			input = getInput().get(0);
			dout1  = getInput().get(1);
		}
		
		long ret = -1;
		if(dimString.equals("K") && filter != null) {
			ret = getNonNegative(ret, getNonNegative(_cachedParams.K, filter.getDim1()));
		}
		else if(dimString.equals("CRS") && filter != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(_cachedParams.C, _cachedParams.R, _cachedParams.S), filter.getDim2()));
		}
		else if(dimString.equals("N") && input != null) {
			ret = getNonNegative(ret, getNonNegative(_cachedParams.N, input.getDim1()));
		}
		else if(dimString.equals("CHW") && input != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(_cachedParams.C, _cachedParams.H, _cachedParams.W), input.getDim2()));
		}
		else if(dimString.equals("N") && dout != null) {
			ret = getNonNegative(ret, getNonNegative(_cachedParams.N, dout.getDim1()));
		}
		else if(dimString.equals("KPQ") && dout != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(_cachedParams.K, _cachedParams.P, _cachedParams.Q), dout.getDim2()));
		}
		else if(dimString.equals("N") && dout1 != null) {
			ret = getNonNegative(ret, getNonNegative(_cachedParams.N, dout1.getDim1()));
		}
		else if(dimString.equals("CPQ") && dout1 != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(_cachedParams.C, _cachedParams.P, _cachedParams.Q), dout1.getDim2()));
		}
		else if(dimString.equals("K")) {
			ret = getNonNegative(ret, _cachedParams.K >= 0 ? _cachedParams.K : -1);
		}
		else if(dimString.equals("CRS")) {
			ret = getNonNegative(ret, nonNegativeMultiply(_cachedParams.C, _cachedParams.R, _cachedParams.S));
		}
		else if(dimString.equals("N")) {
			ret = getNonNegative(ret, _cachedParams.N >= 0 ? _cachedParams.N : -1);
		}
		else if(dimString.equals("CHW")) {
			ret = getNonNegative(ret, nonNegativeMultiply(_cachedParams.C, _cachedParams.H, _cachedParams.W));
		}
		else if(dimString.equals("KPQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(_cachedParams.K, _cachedParams.P, _cachedParams.Q));
		}
		else if(dimString.equals("PQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(_cachedParams.P, _cachedParams.Q));
		}
		else if(dimString.equals("CPQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(_cachedParams.C, _cachedParams.P, _cachedParams.Q));
		}
		else {
			throw new RuntimeException("Unsupported dimension:" + dimString + " for operator " + getOp().name());
		}
		
		if(LOG.isDebugEnabled() && ret < 0) {
			LOG.debug("Unknown dimension " + dimString + " for DnnOp:" + op.name() + 
					" img_dim=[" + _cachedParams.N + " " + _cachedParams.C + " " + _cachedParams.H + " " + _cachedParams.W + "]" +
					" filter_dim=[" + _cachedParams.K + " " + _cachedParams.C + " " + _cachedParams.R + " " + _cachedParams.S + "]" + 
					" output_feature_map=[" + _cachedParams.P + " " + _cachedParams.Q + "] stride=[" + _cachedParams.stride_h + " " + _cachedParams.stride_w + "]" +
					" pad=[" + _cachedParams.pad_h + " " + _cachedParams.pad_w + "]");
		}
		return ret;
	}
	
	private static long nonNegativeMultiply(long val1, long val2, long val3) {
		if(val1 >= 0 && val2 >= 0 && val3 >= 0) {
			return val1 * val2 * val3;
		}
		else return -1;
	}
	private static long nonNegativeMultiply(long val1, long val2) {
		if(val1 >= 0 && val2 >= 0) {
			return val1 * val2;
		}
		else return -1;
	}
	private static long getNonNegative(long val1, long val2) {
		if(val1 >= 0 && val2 >= 0) {
			if(val1 == val2) return val1;
			else throw new RuntimeException("Incorrect dimensions in DnnOp: " + val1 + " != " + val2);
		}
		else if(val1 >= 0) return val1;
		else if(val2 >= 0) return val2;
		else return -1;
	}
	// ------------------------------------------------------------------------------------------------------
}
