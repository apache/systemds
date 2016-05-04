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

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.ConvolutionTransform;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class ConvolutionOp extends Hop 
{	
	private Hop.ConvOp op;
	
	long N = -1; long C = -1; long H = -1; long W = -1;
	long K = -1; long R = -1; long S = -1;
	long stride1 = -1;
	long stride2 = -1;
	long padding1 = -1;
	long padding2 = -1;
	long P = -1; long Q = -1;

	private ConvolutionOp() {
		//default constructor for clone
	}
	
	public ConvolutionOp(String l, DataType dt, ValueType vt, ConvOp o, Hop inp)
	{
		super(l, dt, vt);
		op = o;
		getInput().add(0, inp);
		inp.getParent().add(this);
		
		//compute unknown dims and nnz
		refreshSizeInformation();
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
			case IM2COL:
			case RESHAPE_COL:
			case ROTATE180:
			case COL2IM:
			case MAX_POOLING:
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D:
			case DIRECT_CONV2D_BACKWARD_DATA:
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{	
				if( et == ExecType.CP )
				{
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
		
		if(inputs.size() != expectedNumInputs) {
			throw new HopsException("Incorrect number of inputs for " + op.name());
		}
		
		Lop in = inputs.get(0).constructLops();
		ConvolutionTransform transform1 = new ConvolutionTransform( in, 
				HopsConv2Lops.get(op), getDataType(), getValueType(), et);
		setOutputDimensions(transform1);
		setLineNumbers(transform1);
		
		// stride1, stride2, padding1, padding2  
		// input_shape1, input_shape2, input_shape3, input_shape4, 
		// filter_shape1, filter_shape2, filter_shape3, filter_shape4
		for( int i=1; i <= (expectedNumInputs-1); i++ )
		{
			Lop ltmp = inputs.get(i).constructLops();
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
		switch(op) 
		{
			case RESHAPE_COL:
			case ROTATE180:
			{
				sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				break;
			}
			case IM2COL:
			case COL2IM: 
			case MAX_POOLING: 
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D: 
			case DIRECT_CONV2D_BACKWARD_FILTER: 
			case DIRECT_CONV2D_BACKWARD_DATA:
				sparsity = 1.0; // worst-case estimate
				break;
		}
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
	
		Hop input1 = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input1);
			
		switch(op) 
		{
			case RESHAPE_COL:
			{
				parseInput();
				ret = new long[3];
				ret[0] = N;
				ret[1] = getExtractedVal(K, P, Q);;
				ret[2] = mc.getNonZeros(); // exact estimates
				break;
			}
			case ROTATE180:
			{
				parseInput();
				ret = new long[3];
				ret[0] = getExtractedVal(N, P, Q);
				ret[1] = K;
				ret[2] = mc.getNonZeros(); // exact estimates
				break;
			}
			case IM2COL:
			case COL2IM: 
			case MAX_POOLING: 
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D: 
			case DIRECT_CONV2D_BACKWARD_FILTER: 
			case DIRECT_CONV2D_BACKWARD_DATA:
				break;
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
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
			{
				_etype = ExecType.CP;
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
	
		return _etype;
	}
	
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4
	void parseInput() {
		// For pooling: x,stride,stride,pad,pad,numImg,numChannels,imgSize,imgSize,1,1,poolSize1,poolSize2)
		stride1 = extractValue(getInput().get(1));
		stride2 = extractValue(getInput().get(2));
		padding1 = extractValue(getInput().get(3));
		padding2 = extractValue(getInput().get(4));
		N = extractValue(getInput().get(5));
		C = extractValue(getInput().get(6));
		H = extractValue(getInput().get(7));
		W = extractValue(getInput().get(8));
		K = extractValue(getInput().get(9));
		C = (C <= 0) ? extractValue(getInput().get(10)) : C;
		R = extractValue(getInput().get(11));
		S = extractValue(getInput().get(12));
		if(H > 0 && R > 0 && stride1 > 0 && padding1 > 0)
			P = ConvolutionUtils.getP(H, R, stride1, padding1);
		if(W > 0 && S > 0 && stride2 > 0 && padding2 > 0)
		Q = ConvolutionUtils.getQ(W, S, stride2, padding2);
	}
	
	long getExtractedVal(long val1, long val2) {
		if(val1 == -1 || val2 == -1) {
			return -1;
		}
		return val1*val2;
	}
	
	long getExtractedVal(long val1, long val2, long val3) {
		if(val1 == -1 || val2 == -1 || val3 == -1) {
			return -1;
		}
		return val1*val2*val3;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		
		switch(op) 
		{
			case IM2COL:
			{
				parseInput();
				_dim1 = getExtractedVal(C, R, S);
				_dim2 = getExtractedVal(N, P, Q);
				_nnz = -1;
				break;
			}
			case COL2IM:
			{
				parseInput();
				// Set _dim1, _dim2 and if possible _nnz (use input1.getNnz())
				_dim1 = N;
				_dim2 = getExtractedVal(C, H, W);
				_nnz = -1; // cannot infer stats
				break;
			}
			case RESHAPE_COL:
			{
				parseInput();
				_dim1 = N;
				_dim2 = getExtractedVal(K, P, Q);
				_nnz = input1.getNnz(); // exact estimates
				break;
			}
			case ROTATE180:
			{
				parseInput();
				_dim1 = getExtractedVal(N, P, Q);
				_dim2 = K;
				_nnz = input1.getNnz(); // exact estimates
				break;
			}
			case MAX_POOLING:
			{
				parseInput();
				_dim1 = N;
				_dim2 = getExtractedVal(C, P, Q);
				_nnz = -1; // cannot infer stats
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				parseInput();
				_dim1 = N;
				_dim2 = getExtractedVal(C, H, W);
				_nnz = -1;
				break;
			}
			case DIRECT_CONV2D:
			{
				parseInput();
				_dim1 = N;
				_dim2 = getExtractedVal(K, P, Q);
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				parseInput();
				_dim1 = N;
				_dim2 = getExtractedVal(C, H, W);
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				parseInput();
				_dim1 = K;
				_dim2 = getExtractedVal(C, R, S);
				_nnz = -1; // cannot infer stats
				break;
			}
			default:
				throw new RuntimeException("The sizes are not refreshed for " + op.name());
		}
	}
	
	private long extractValue(Hop hop)  {
		if(hop instanceof LiteralOp)
			return (long) HopRewriteUtils.getDoubleValueSafe((LiteralOp)hop);
		return -1;
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ConvolutionOp ret = new ConvolutionOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof ConvolutionOp) )
			return false;
		
		ConvolutionOp that2 = (ConvolutionOp)that;		
		boolean ret =  (op == that2.op)
				    && (getInput().size()==that.getInput().size());
				
		//compare all childs (see reshape, sort)
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
}