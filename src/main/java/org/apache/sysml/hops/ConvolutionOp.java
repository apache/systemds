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

package org.apache.sysml.hops;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.lops.ConvolutionTransform;
import org.apache.sysml.lops.ConvolutionTransform.OperationTypes;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;

import java.util.ArrayList;

public class ConvolutionOp extends Hop  implements MultiThreadedHop
{	
	private Hop.ConvOp op;

	private int _maxNumThreads = -1; //-1 for unlimited

	private ConvolutionOp() {
		//default constructor for clone
	}

	public ConvolutionOp(String l, DataType dt, ValueType vt, ConvOp o, ArrayList<Hop> inp) 
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

	@Override
	public void checkArity() throws HopsException {
		HopsException.check(_input.size() >= 1, this, "should have at least one input but has %d inputs", _input.size());
	}

	public ConvOp getOp()
	{
		return op;
	}
	
	@Override
	public String getOpString() {
		return "" + HopsConv2Lops.get(op);
	}

	private boolean isEligibleForSpark() {
		// return (op == ConvOp.DIRECT_CONV2D || op == ConvOp.MAX_POOLING) ? true : false;
		return false;
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		return true;
	}
	
	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();
		
		ExecType et = optFindExecType();
		
		ArrayList<Hop> inputs = getInput();
		switch( op )
		{
			case MAX_POOLING:
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D:
			case DIRECT_CONV2D_BACKWARD_DATA:
			case DIRECT_CONV2D_BACKWARD_FILTER:
			case BIAS_ADD:
			case BIAS_MULTIPLY:
			{	
				if(et == ExecType.CP || et == ExecType.GPU) {
					setLops(constructConvolutionLops(et, inputs));
					break;
				}
				else {
					throw new HopsException("Unimplemented ConvolutionOp for execution type: " + et.name());
				}
				// break;
			}
			default: 
				throw new HopsException("Unsupported lops construction for operation type '"+op+"'.");
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
				
		return getLops();
	}
	
	public void setOp(ConvOp op) {
		this.op = op;
	}
	
	private int getNumExpectedInputs() {
		switch(op) {
			case MAX_POOLING_BACKWARD: 
			case DIRECT_CONV2D:
			case DIRECT_CONV2D_BACKWARD_FILTER:
			case DIRECT_CONV2D_BACKWARD_DATA:
				return 14;
			case BIAS_ADD:
			case BIAS_MULTIPLY:
				return 2;
			default:
				return 13;
		}
	}
	
	private boolean isInputReLU(Hop input) {
		return input instanceof UnaryOp && ((UnaryOp) input).getOp() == OpOp1.SELP;
	}
	
	private boolean isInputConv2d(Hop input) {
		return input instanceof ConvolutionOp && ((ConvolutionOp) input).getOp() == ConvOp.DIRECT_CONV2D;
	}
	
	public Lop constructConvolutionLops(ExecType et, ArrayList<Hop> inputs) throws HopsException, LopsException {
		if(inputs.size() != getNumExpectedInputs()) 
			throw new HopsException("Incorrect number of inputs for " + op.name());
		
		Lop in = null; Lop in2 = null;
		ArrayList<Hop> inputs1 = inputs;
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		OperationTypes lopOp = HopsConv2Lops.get(op);

		// RELU_MAX_POOLING and RELU_MAX_POOLING_BACKWARD is extremely useful for CP backend 
		// by reducing unnecessary sparse-to-dense-to-sparse conversion.
		// For other backends, this operators is not necessary as it reduces an additional relu operator.
		if(OptimizerUtils.ALLOW_OPERATOR_FUSION && et == ExecType.CP && op == ConvOp.MAX_POOLING && isInputReLU(inputs.get(0))) {
			in = inputs.get(0).getInput().get(0).constructLops();
			lopOp = OperationTypes.RELU_MAX_POOLING;
		}
		else if(OptimizerUtils.ALLOW_OPERATOR_FUSION && et == ExecType.CP && op == ConvOp.MAX_POOLING_BACKWARD && isInputReLU(inputs.get(0))) {
			in = inputs.get(0).getInput().get(0).constructLops();
			lopOp = OperationTypes.RELU_MAX_POOLING_BACKWARD;
		}
		else if(OptimizerUtils.ALLOW_OPERATOR_FUSION && op == ConvOp.BIAS_ADD && isInputConv2d(inputs.get(0))) {
			lopOp = OperationTypes.DIRECT_CONV2D_BIAS_ADD;
			
			// the first lop is image 
			in = inputs.get(0).getInput().get(0).constructLops();
			// the second lop is bias
			in2 = inputs.get(1).constructLops();
			
			// Use the inputs from conv2d rather than bias_add
			inputs1 = inputs.get(0).getInput();
		}
		else {
			in = inputs.get(0).constructLops();
		}
		
//		// TODO: Inserting reblock requires knowing columns apriori
//		ConvolutionTransform transform1 = new ConvolutionTransform(addReblockIfNecessary(et, lopOp, in), lopOp, getDataType(), getValueType(), et, k);
//		setReblockedOutputDimension(et, transform1);
		ConvolutionTransform transform1 = new ConvolutionTransform(in, lopOp, getDataType(), getValueType(), et, k, computeIntermediateMemEstimate(-1, -1, -1 ));
		setOutputDimensions(transform1);
		
		setLineNumbers(transform1);
		in.addOutput(transform1);
		
		if(in2 != null) {
			transform1.addInput(in2);
			in2.addOutput(transform1);
		}
		
		// stride1, stride2, padding1, padding2  
		// input_shape1, input_shape2, input_shape3, input_shape4, 
		// filter_shape1, filter_shape2, filter_shape3, filter_shape4
		for( int i=1; i < inputs1.size(); i++ )
		{
			Lop ltmp = inputs1.get(i).constructLops();
			transform1.addInput(ltmp);
			ltmp.addOutput(transform1);
		}
		transform1.setLevel(); //force order of added lops
		return transform1;
	}

			
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double sparsity = 1.0;
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	// ---------------------------------------------------------------
	// Utility methods to guard the computation of memory estimates in presense of unknowns
	private static class IntermediateDimensions {
		int dim1; int dim2; double sp;
		public IntermediateDimensions(ConvolutionOp h, String dim1Str, String dim2Str, double sp) {
			dim1 = (int) h.getDim(dim1Str);
			dim2 = (int) h.getDim(dim2Str);
			this.sp = sp;
		}
		public IntermediateDimensions(ConvolutionOp h, String dim1Str, String dim2Str) {
			dim1 = (int) h.getDim(dim1Str);
			dim2 = (int) h.getDim(dim2Str);
			sp = 1;
		}
		public IntermediateDimensions(ConvolutionOp h, int dim1, String dim2Str) {
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
		ArrayList<IntermediateDimensions> gpuIntermediates = new ArrayList<IntermediateDimensions>();
		ArrayList<IntermediateDimensions> cpIntermediates = new ArrayList<IntermediateDimensions>();
		if(getOp() == ConvOp.DIRECT_CONV2D) {
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
		else if(getOp() == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
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
		else if(getOp() == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER) {
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
		else if(getOp() == ConvOp.MAX_POOLING) {
			// Account for potential sparse-to-dense conversion of atleast 1 input row
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
		}
		else if(getOp() == ConvOp.MAX_POOLING_BACKWARD) {
			// Account for potential sparse-to-dense conversion of atleast 1 input + dout row
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CHW"));
			gpuIntermediates.add(new IntermediateDimensions(this, 1, "CPQ"));
		}
		
		if(gpuIntermediates.size() > 0 || cpIntermediates.size() > 0)
			return computeIntermediateMemEstimateHelper(gpuIntermediates, cpIntermediates);
		else
			return 0;
	}
	
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		// [numRows, numCols, NNZ] 
		long[] ret = new long[3];
		
		if(op == ConvOp.BIAS_ADD || op == ConvOp.BIAS_MULTIPLY) {
			MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
			ret[0] = mc[0].rowsKnown() ? mc[0].getRows() : -1;
			ret[1] = mc[0].colsKnown() ? mc[0].getCols() : -1;
			ret[2] = -1;
			return (ret[0]>0 && ret[1]>0) ? ret : null;
		}
		
		refreshSizeInformation();
		ret[0] = _dim1; ret[1] = _dim2; ret[2] = _nnz;
		
		//safe return (create entry only if at least dims known)
		return (ret[0]>0 && ret[1]>0) ? ret : null;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else {
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		// TODO: Fix this after adding remaining spark instructions
		_etype = !isEligibleForSpark() && _etype == REMOTE ?  ExecType.CP : _etype;
		
		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		return _etype;
	}
	
	// Caching parameters speed-ups dynamic recompilation time by avoiding unnecessary computeSizeInformation
	private ConvolutionParameters _cachedParams = new ConvolutionParameters(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, _maxNumThreads);
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4
	ConvolutionParameters parseInput() throws DMLRuntimeException {
		if(op == ConvOp.MAX_POOLING_BACKWARD 
				|| op == ConvOp.DIRECT_CONV2D 
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
			_cachedParams.setIfUnknown(
					getInput().get(6),
					getInput().get(7), 
					getInput().get(8), 
					getInput().get(9), 
					getInput().get(10), 
					getInput().get(12), 
					getInput().get(13), 
					getInput().get(2), 
					getInput().get(3), 
					getInput().get(4), 
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
		return _cachedParams;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if(op == ConvOp.BIAS_ADD || op == ConvOp.BIAS_MULTIPLY) {
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			_nnz = -1; // cannot infer stats
			return;
		}
		
		switch(op) 
		{
			case MAX_POOLING:
			{	
				_dim1 = getDim("N");
				_dim2 = getDim("CPQ");
				_nnz = -1; // cannot infer stats
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				_dim1 = getDim("N");
				_dim2 = getDim("CHW");
				_nnz = -1;
				break;
			}
			case DIRECT_CONV2D:
			{
				_dim1 = getDim("N");
				_dim2 = getDim("KPQ");
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				_dim1 = getDim("N");
				_dim2 = getDim("CHW");
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				_dim1 = getDim("K");
				_dim2 = getDim("CRS");
				_nnz = -1; // cannot infer stats
				break;
			}
			default:
				throw new RuntimeException("The sizes are not refreshed for " + op.name());
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ConvolutionOp ret = new ConvolutionOp();	
		
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
		if( !(that instanceof ConvolutionOp) )
			return false;
		
		ConvolutionOp that2 = (ConvolutionOp)that;
		
		boolean ret =  (op == that2.op)
				    && (getInput().size()==that.getInput().size())
				    && _maxNumThreads == that2._maxNumThreads;
		
		//compare all childs
		if( ret ) //sizes matched
			for( int i=0; i<_input.size(); i++ )
				ret &= getInput().get(i) == that2.getInput().get(i);
		
		return ret;
	}

	@Override
	public void setMaxNumThreads( int k ) {
		_maxNumThreads = k;
	}
	
	@Override
	public int getMaxNumThreads() {
		return _maxNumThreads;
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
		if(op == ConvOp.BIAS_ADD || op == ConvOp.BIAS_MULTIPLY) {
			throw new RuntimeException("getDim method should not be invoked for bias_add and bias_multiply");
		}
		ConvolutionParameters params;
		try {
			params = parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		Hop filter = null; 	// shape: K x CRS 
		Hop input = null; 	// shape: N x CHW
		Hop dout = null;	// shape: N x KPQ
		Hop dout1 = null;	// shape: N x CPQ
		
		if(getOp() == ConvOp.DIRECT_CONV2D) {
			input  = getInput().get(0);
			filter = getInput().get(1);
		}
		else if(getOp() == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
			filter = getInput().get(0);
			dout  = getInput().get(1);
		}
		else if(getOp() == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER) {
			input = getInput().get(0);
			dout  = getInput().get(1);
		}
		else if(getOp() == ConvOp.MAX_POOLING) {
			input = getInput().get(0);
		}
		else if(getOp() == ConvOp.MAX_POOLING_BACKWARD) {
			input = getInput().get(0);
			dout1  = getInput().get(1);
		}
		
		long ret = -1;
		if(dimString.equals("K") && filter != null) {
			ret = getNonNegative(ret, getNonNegative(params.K, filter._dim1));
		}
		else if(dimString.equals("CRS") && filter != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(params.C, params.R, params.S), filter._dim2));
		}
		else if(dimString.equals("N") && input != null) {
			ret = getNonNegative(ret, getNonNegative(params.N, input._dim1));
		}
		else if(dimString.equals("CHW") && input != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(params.C, params.H, params.W), input._dim2));
		}
		else if(dimString.equals("N") && dout != null) {
			ret = getNonNegative(ret, getNonNegative(params.N, dout._dim1));
		}
		else if(dimString.equals("KPQ") && dout != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(params.K, params.P, params.Q), dout._dim2));
		}
		else if(dimString.equals("N") && dout1 != null) {
			ret = getNonNegative(ret, getNonNegative(params.N, dout1._dim1));
		}
		else if(dimString.equals("CPQ") && dout1 != null) {
			ret = getNonNegative(ret, getNonNegative(nonNegativeMultiply(params.C, params.P, params.Q), dout1._dim2));
		}
		else if(dimString.equals("K")) {
			ret = getNonNegative(ret, params.K >= 0 ? params.K : -1);
		}
		else if(dimString.equals("CRS")) {
			ret = getNonNegative(ret, nonNegativeMultiply(params.C, params.R, params.S));
		}
		else if(dimString.equals("N")) {
			ret = getNonNegative(ret, params.N >= 0 ? params.N : -1);
		}
		else if(dimString.equals("CHW")) {
			ret = getNonNegative(ret, nonNegativeMultiply(params.C, params.H, params.W));
		}
		else if(dimString.equals("KPQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(params.K, params.P, params.Q));
		}
		else if(dimString.equals("PQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(params.P, params.Q));
		}
		else if(dimString.equals("CPQ")) {
			ret = getNonNegative(ret, nonNegativeMultiply(params.C, params.P, params.Q));
		}
		else {
			throw new RuntimeException("Unsupported dimension:" + dimString + " for operator " + getOp().name());
		}
		
		if(LOG.isDebugEnabled() && ret < 0) {
			LOG.debug("Unknown dimension " + dimString + " for ConvolutionOp:" + op.name() + 
					" img_dim=[" + params.N + " " + params.C + " " + params.H + " " + params.W + "]" +
					" filter_dim=[" + params.K + " " + params.C + " " + params.H + " " + params.W + "]" + 
					" output_feature_map=[" + params.P + " " + params.Q + "] stride=[" + params.stride_h + " " + params.stride_w + "]" +
					" pad=[" + params.pad_h + " " + params.pad_w + "]");
		}
		return ret;
	}
	
	private long nonNegativeMultiply(long val1, long val2, long val3) {
		if(val1 >= 0 && val2 >= 0 && val3 >= 0) {
			return val1 * val2 * val3;
		}
		else return -1;
	}
	private long nonNegativeMultiply(long val1, long val2) {
		if(val1 >= 0 && val2 >= 0) {
			return val1 * val2;
		}
		else return -1;
	}
	private long getNonNegative(long val1, long val2) {
		if(val1 >= 0 && val2 >= 0) {
			if(val1 == val2) return val1;
			else throw new RuntimeException("Incorrect dimensions in Convolution Hop: " + val1 + " != " + val2);
		}
		else if(val1 >= 0) return val1;
		else if(val2 >= 0) return val2;
		else return -1;
	}
	// ------------------------------------------------------------------------------------------------------
}
