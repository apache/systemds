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

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.lops.ConvolutionTransform;
import org.apache.sysml.lops.ConvolutionTransform.OperationTypes;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;

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

	public ConvOp getOp()
	{
		return op;
	}
	
	@Override
	public String getOpString() {
		return "" + HopsConv2Lops.get(op);
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
			{	
				//TODO: Fix me. Currently forcing the instruction to GPU if gpu flag is set
				if(DMLScript.USE_ACCELERATOR) {
					et = ExecType.GPU;
					setLops(constructConvolutionLops(et, inputs));
					break;
				}
				else if(et == ExecType.CP) {
					setLops(constructConvolutionLops(et, inputs));
					break;
				}			
				else {
					// TODO: Add support for SPARK/MR backends once we are happy with the performance of
					// single node Lenet script. 
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

	public Lop constructConvolutionLops(ExecType et, ArrayList<Hop> inputs) throws HopsException, LopsException {
		int expectedNumInputs = 13;
		if(op == ConvOp.MAX_POOLING_BACKWARD 
				|| op == ConvOp.DIRECT_CONV2D 
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
			expectedNumInputs = 14;
		}
		else if(op == ConvOp.BIAS_ADD) {
			expectedNumInputs = 2;
		}
		
		if(inputs.size() != expectedNumInputs) {
			throw new HopsException("Incorrect number of inputs for " + op.name());
		}
		
		Lop in = null; Lop in2 = null;
		OperationTypes lopOp = HopsConv2Lops.get(op);
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		ArrayList<Hop> inputs1 = inputs;
		if(op == ConvOp.MAX_POOLING && et == ExecType.CP && inputs.get(0) instanceof UnaryOp
				&& ((UnaryOp) inputs.get(0)).getOp() == OpOp1.SELP) {
			in = inputs.get(0).getInput().get(0).constructLops();
			lopOp = OperationTypes.RELU_MAX_POOLING;
		}
		else if(op == ConvOp.BIAS_ADD && et == ExecType.CP && inputs.get(0) instanceof ConvolutionOp
				&& ((ConvolutionOp) inputs.get(0)).getOp() == ConvOp.DIRECT_CONV2D) {
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
		ConvolutionTransform transform1 = new ConvolutionTransform( in, lopOp, getDataType(), getValueType(), et, k);
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
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{	
		//default: no intermediate memory requirements
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		// [numRows, numCols, NNZ] 
		long[] ret = null;
		
		if(op == ConvOp.BIAS_ADD) {
			MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
			ret = new long[3];
			ret[0] = mc[0].rowsKnown() ? mc[0].getRows() : -1;
			ret[1] = mc[0].colsKnown() ? mc[0].getCols() : -1;
			ret[2] = -1;
			return ret;
		}
	
		ConvolutionParameters params;
		try {
			params = parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		switch(op) 
		{
			case MAX_POOLING:
			{
				ret = new long[3];
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getExtractedVal(params.C, params.P, params.Q);
				ret[2] = -1;
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				ret = new long[3];
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getInput().get(0)._dim2;
				ret[2] = -1;
				break;
			}
			case DIRECT_CONV2D:
			{
				ret = new long[3];
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getExtractedVal(getInput().get(1)._dim1, params.P, params.Q);
				ret[2] = -1;
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				ret = new long[3];
				ret[0] = getInput().get(1)._dim1;
				ret[1] = getInput().get(1)._dim2;
				ret[2] = -1;
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				ret = new long[3];
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getInput().get(0)._dim2;
				ret[2] = -1;
				break;
			}
			default:
				throw new RuntimeException("Unsupported op:" + op.name());
		}
		
		if(LOG.isDebugEnabled() && (ret[0] <= 0 || ret[1] <= 0)) {
			LOG.debug("Unknown dimensions for ConvolutionOp in inferOutputCharacteristics:" + op.name() + " " + ret[0] + " " + ret[1] + 
					" img_dim=[" + params.N + " " + params.C + " " + params.H + " " + params.W + "]" +
					" filter_dim=[" + params.K + " " + params.C + " " + params.H + " " + params.W + "]" + 
					" output_feature_map=[" + params.P + " " + params.Q + "] stride=[" + params.stride_h + " " + params.stride_w + "]" +
					" pad=[" + params.pad_h + " " + params.pad_w + "]");
		}
		
		return ret;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		//TODO: Remove this once memEstimate is fixed for these instructions 
		if((op == ConvOp.MAX_POOLING || op == ConvOp.MAX_POOLING_BACKWARD) && DMLScript.USE_ACCELERATOR) {
			return ExecType.GPU;
		}
	
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{	
			// TODO: After adding Spark backend, uncomment this
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else 
			{
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
		
		_etype = ExecType.CP;
	
		return _etype;
	}
	
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4
	ConvolutionParameters parseInput() throws DMLRuntimeException {
		ConvolutionParameters params = null;
		if(op == ConvOp.MAX_POOLING_BACKWARD 
				|| op == ConvOp.DIRECT_CONV2D 
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
			params = new ConvolutionParameters(
					computeSizeInformation(getInput().get(6)),
					computeSizeInformation(getInput().get(7)), 
					computeSizeInformation(getInput().get(8)), 
					computeSizeInformation(getInput().get(9)), 
					computeSizeInformation(getInput().get(10)), 
					computeSizeInformation(getInput().get(12)), 
					computeSizeInformation(getInput().get(13)), 
					computeSizeInformation(getInput().get(2)), 
					computeSizeInformation(getInput().get(3)), 
					computeSizeInformation(getInput().get(4)), 
					computeSizeInformation(getInput().get(5)), _maxNumThreads);
		}
		else {
			params = new ConvolutionParameters(
					computeSizeInformation(getInput().get(5)),
					computeSizeInformation(getInput().get(6)), 
					computeSizeInformation(getInput().get(7)), 
					computeSizeInformation(getInput().get(8)), 
					computeSizeInformation(getInput().get(9)), 
					computeSizeInformation(getInput().get(11)), 
					computeSizeInformation(getInput().get(12)), 
					computeSizeInformation(getInput().get(1)), 
					computeSizeInformation(getInput().get(2)), 
					computeSizeInformation(getInput().get(3)), 
					computeSizeInformation(getInput().get(4)), _maxNumThreads);
		}
		return params;
	}

	public static long getExtractedVal(long val1, long val2, long val3) {
		if(val1 == -1 || val2 == -1 || val3 == -1) {
			return -1;
		}
		return val1*val2*val3;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if(op == ConvOp.BIAS_ADD) {
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			return;
		}
		
		ConvolutionParameters params;
		try {
			params = parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		switch(op) 
		{
			case MAX_POOLING:
			{	
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getExtractedVal(params.C, params.P, params.Q);
				_nnz = -1; // cannot infer stats
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getInput().get(0)._dim2;
				_nnz = -1;
				break;
			}
			case DIRECT_CONV2D:
			{
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getExtractedVal(getInput().get(1)._dim1, params.P, params.Q);
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getInput().get(0)._dim2;
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				_dim1 = getInput().get(1)._dim1;
				_dim2 = getInput().get(1)._dim2;
				_nnz = -1; // cannot infer stats
				break;
			}
			default:
				throw new RuntimeException("The sizes are not refreshed for " + op.name());
		}
		
		if(LOG.isDebugEnabled() && (_dim1 <= 0 || _dim2 <= 0)) {
			LOG.debug("Unknown dimensions for ConvolutionOp in refreshSizeInformation:" + op.name() + " " + _dim1 + " " + _dim2 + 
					" img_dim=[" + params.N + " " + params.C + " " + params.H + " " + params.W + "]" +
					" filter_dim=[" + params.K + " " + params.C + " " + params.H + " " + params.W + "]" + 
					" output_feature_map=[" + params.P + " " + params.Q + "] stride=[" + params.stride_h + " " + params.stride_w + "]" +
					" pad=[" + params.pad_h + " " + params.pad_w + "]");
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
	public void printMe() throws HopsException 
	{
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	public void setMaxNumThreads( int k ) {
		_maxNumThreads = k;
	}
	
	@Override
	public int getMaxNumThreads() {
		return _maxNumThreads;
	}
}
