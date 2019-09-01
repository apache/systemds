/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.hops;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.lops.CentralMoment;
import org.tugraz.sysds.lops.CoVariance;
import org.tugraz.sysds.lops.Ctable;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.LopsException;
import org.tugraz.sysds.lops.PickByCount;
import org.tugraz.sysds.lops.SortKeys;
import org.tugraz.sysds.lops.Ternary;
import org.tugraz.sysds.parser.Statement;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

/** Primary use cases for now, are
 * 		quantile (<n-1-matrix>, <n-1-matrix>, <literal>):      quantile (A, w, 0.5)
 * 		quantile (<n-1-matrix>, <n-1-matrix>, <scalar>):       quantile (A, w, s)
 * 		interquantile (<n-1-matrix>, <n-1-matrix>, <scalar>):  interquantile (A, w, s)
 * 
 * Keep in mind, that we also have binaries for it w/o weights.
 * 	quantile (A, 0.5)
 * 	quantile (A, s)
 * 	interquantile (A, s)
 * 
 * Note: this hop should be called AggTernaryOp in consistency with AggUnaryOp and AggBinaryOp;
 * however, since there does not exist a real TernaryOp yet - we can leave it as is for now.
 *
 * CTABLE op takes 2 extra inputs with target dimensions for padding and pruning.
 */
public class TernaryOp extends Hop 
{
	
	public static boolean ALLOW_CTABLE_SEQUENCE_REWRITES = true;
	
	private OpOp3 _op = null;
	
	//ctable specific flags 
	// flag to indicate the existence of additional inputs representing output dimensions
	private boolean _dimInputsPresent = false;
	private boolean _disjointInputs = false;
	
	
	private TernaryOp() {
		//default constructor for clone
	}
	
	public TernaryOp(String l, DataType dt, ValueType vt, Hop.OpOp3 o,
			Hop inp1, Hop inp2, Hop inp3) {
		super(l, dt, vt);
		_op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);
		getInput().add(2, inp3);
		inp1.getParent().add(this);
		inp2.getParent().add(this);
		inp3.getParent().add(this);
	}
	
	// Constructor the case where TertiaryOp (table, in particular) has
	// output dimensions
	public TernaryOp(String l, DataType dt, ValueType vt, Hop.OpOp3 o,
			Hop inp1, Hop inp2, Hop inp3, Hop inp4, Hop inp5) {
		super(l, dt, vt);
		_op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);
		getInput().add(2, inp3);
		getInput().add(3, inp4);
		getInput().add(4, inp5);
		inp1.getParent().add(this);
		inp2.getParent().add(this);
		inp3.getParent().add(this);
		inp4.getParent().add(this);
		inp5.getParent().add(this);
		_dimInputsPresent = true;
	}

	@Override
	public void checkArity() {
		int sz = _input.size();
		if (_dimInputsPresent) {
			// only CTABLE
			HopsException.check(sz == 5, this, "should have arity 5 for op %s but has arity %d", _op, sz);
		} else {
			HopsException.check(sz == 3, this, "should have arity 3 for op %s but has arity %d", _op, sz);
		}
	}

	public OpOp3 getOp(){
		return _op;
	}
	
	public void setDisjointInputs(boolean flag){
		_disjointInputs = flag;
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		switch( _op ) {
			case MOMENT:
			case COV:
			case CTABLE:
			case INTERQUANTILE:
			case QUANTILE:
			case IFELSE:
				return false;
			case MINUS_MULT:
			case PLUS_MULT:
				return true;
			default:
				throw new RuntimeException("Unsupported operator:" + _op.name());
		}
	}
	
	@Override
	public Lop constructLops() 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		try 
		{
			switch( _op ) {
				case MOMENT:
					constructLopsCentralMoment();
					break;
					
				case COV:
					constructLopsCovariance();
					break;
					
				case QUANTILE:
				case INTERQUANTILE:
					constructLopsQuantile();
					break;
					
				case CTABLE:
					constructLopsCtable();
					break;
				
				case PLUS_MULT:
				case MINUS_MULT:
				case IFELSE:
					constructLopsTernaryDefault();
					break;
					
				default:
					throw new HopsException(this.printErrorLocation() + "Unknown TernaryOp (" + _op + ") while constructing Lops \n");

			}
		} 
		catch(LopsException e) {
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for TernaryOp Hop " , e);
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
		
		return getLops();
	}

	/**
	 * Method to construct LOPs when op = CENTRAILMOMENT.
	 */
	private void constructLopsCentralMoment()
	{	
		if ( _op != OpOp3.MOMENT )
			throw new HopsException("Unexpected operation: " + _op + ", expecting " + OpOp3.MOMENT );
		
		ExecType et = optFindExecType();
		
		CentralMoment cm = new CentralMoment(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getDataType(), getValueType(), et);
		cm.getOutputParameters().setDimensions(0, 0, 0, -1);
		setLineNumbers(cm);
		setLops(cm);
	}
	
	/**
	 * Method to construct LOPs when op = COVARIANCE.
	 */
	private void constructLopsCovariance() {
		if ( _op != OpOp3.COV )
			throw new HopsException("Unexpected operation: " + _op + ", expecting " + OpOp3.COV );
		
		ExecType et = optFindExecType();
		
		
		CoVariance cov = new CoVariance(
				getInput().get(0).constructLops(), 
				getInput().get(1).constructLops(), 
				getInput().get(2).constructLops(), 
				getDataType(), getValueType(), et);
		cov.getOutputParameters().setDimensions(0, 0, 0, -1);
		setLineNumbers(cov);
		setLops(cov);
	}
	
	/**
	 * Method to construct LOPs when op = QUANTILE | INTERQUANTILE.
	 */
	private void constructLopsQuantile() {
		
		if ( _op != OpOp3.QUANTILE && _op != OpOp3.INTERQUANTILE )
			throw new HopsException("Unexpected operation: " + _op + ", expecting " + OpOp3.QUANTILE + " or " + OpOp3.INTERQUANTILE );
		
		ExecType et = optFindExecType();
		
		
		SortKeys sort = SortKeys.constructSortByValueLop(
				getInput().get(0).constructLops(), 
				getInput().get(1).constructLops(), 
				SortKeys.OperationTypes.WithWeights, 
				getInput().get(0).getDataType(), getInput().get(0).getValueType(), et);
		PickByCount pick = new PickByCount(
				sort,
				getInput().get(2).constructLops(),
				getDataType(),
				getValueType(),
				(_op == Hop.OpOp3.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
						: PickByCount.OperationTypes.RANGEPICK, et, true);
		sort.getOutputParameters().setDimensions(
				getInput().get(0).getDim1(),
				getInput().get(0).getDim2(),
				getInput().get(0).getBlocksize(),
				getInput().get(0).getNnz());
		
		setOutputDimensions(pick);
		setLineNumbers(pick);
		setLops(pick);
	}

	/**
	 * Method to construct LOPs when op = CTABLE.
	 */
	private void constructLopsCtable() {
		
		if ( _op != OpOp3.CTABLE )
			throw new HopsException("Unexpected operation: " + _op + ", expecting " + OpOp3.CTABLE );
		
		/*
		 * We must handle three different cases: case1 : all three
		 * inputs are vectors (e.g., F=ctable(A,B,W)) case2 : two
		 * vectors and one scalar (e.g., F=ctable(A,B)) case3 : one
		 * vector and two scalars (e.g., F=ctable(A))
		 */

		// identify the particular case
		
		// F=ctable(A,B,W)
		
		DataType dt1 = getInput().get(0).getDataType(); 
		DataType dt2 = getInput().get(1).getDataType(); 
		DataType dt3 = getInput().get(2).getDataType(); 
		Ctable.OperationTypes ternaryOpOrig = Ctable.findCtableOperationByInputDataTypes(dt1, dt2, dt3);
 		
		// Compute lops for all inputs
		Lop[] inputLops = new Lop[getInput().size()];
		for(int i=0; i < getInput().size(); i++) {
			inputLops[i] = getInput().get(i).constructLops();
		}
		
		ExecType et = optFindExecType();
		
		//reset reblock requirement (see MR ctable / construct lops)
		setRequiresReblock( false );
		
		//for CP we support only ctable expand left
		Ctable.OperationTypes ternaryOp = isSequenceRewriteApplicable(true) ? 
			Ctable.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT : ternaryOpOrig;
		boolean ignoreZeros = false;
		
		if( isMatrixIgnoreZeroRewriteApplicable() ) { 
			ignoreZeros = true; //table - rmempty - rshape
			inputLops[0] = ((ParameterizedBuiltinOp)getInput().get(0)).getTargetHop().getInput().get(0).constructLops();
			inputLops[1] = ((ParameterizedBuiltinOp)getInput().get(1)).getTargetHop().getInput().get(0).constructLops();
		}
		
		Ctable ternary = new Ctable(inputLops, ternaryOp, getDataType(), getValueType(), ignoreZeros, et);
		
		ternary.getOutputParameters().setDimensions(_dim1, _dim2, getBlocksize(), -1);
		setLineNumbers(ternary);
		
		//force blocked output in CP and SPARK
		ternary.getOutputParameters().setDimensions(_dim1, _dim2, getBlocksize(), -1);
		
		//ternary opt, w/o reblock in CP
		setLops(ternary);
	}

	private void constructLopsTernaryDefault() {
		ExecType et = optFindExecType();
		if( getInput().stream().allMatch(h -> h.getDataType().isScalar()) )
			et = ExecType.CP; //always CP for pure scalar operations
		Ternary plusmult = new Ternary(HopsOpOp3Lops.get(_op),
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(), 
			getDataType(),getValueType(), et );
		setOutputDimensions(plusmult);
		setLineNumbers(plusmult);
		setLops(plusmult);
	}
	
	@Override
	public String getOpString() {
		String s = new String("");
		s += "t(" + HopsOpOp3String.get(_op) + ")";
		return s;
	}

	@Override
	public boolean allowsAllExecTypes() {
		return true;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		//only quantile and ctable produce matrices
		
		switch( _op ) 
		{
			case CTABLE:
				// since the dimensions of both inputs must be the same, checking for one input is sufficient
				//   worst case dimensions of C = [m,m]
				//   worst case #nnz in C = m => sparsity = 1/m
				// for ctable_histogram also one dimension is known
				double sparsity = OptimizerUtils.getSparsity(dim1, dim2, (nnz<=dim1)?nnz:dim1); 
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
				
			case QUANTILE:
				// This part of the code is executed only when a vector of quantiles are computed
				// Output is a vector of length = #of quantiles to be computed, and it is likely to be dense.
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);
			case PLUS_MULT:
			case MINUS_MULT:
			case IFELSE: {
				if (isGPUEnabled()) {
					// For the GPU, the input is converted to dense
					sparsity = 1.0;
				} else {
					sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				}
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
			}
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		double ret = 0;
		if( _op == OpOp3.CTABLE ) {
			if ( _dim1 >= 0 && _dim2 >= 0 ) {
				// output dimensions are known, and hence a MatrixBlock is allocated
				double sp = OptimizerUtils.getSparsity(_dim1, _dim2, Math.min(nnz, _dim1));
				ret = OptimizerUtils.estimateSizeExactSparsity(_dim1, _dim2, sp );
			}
			else {
				ret =  2*4 * dim1 + //hash table (worst-case overhead 2x)
						  32 * dim1; //values: 2xint,1xObject
			}
		}
		else if ( _op == OpOp3.QUANTILE ) {
			// buffer (=2*input_size) and output (=2*input_size) for SORT operation
			// getMemEstimate works for both cases of known dims and worst-case stats
			ret = getInput().get(0).getMemEstimate() * 4;  
		}
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		DataCharacteristics[] mc = memo.getAllInputStats(getInput());
		
		switch( _op ) 
		{
			case CTABLE:
				boolean dimsSpec = (getInput().size() > 3); 
				
				// Step 1: general dimension info inputs
				long worstCaseDim = -1;
				// since the dimensions of both inputs must be the same, checking for one input is sufficient
				if( mc[0].dimsKnown() || mc[1].dimsKnown() ) {
					// Output dimensions are completely data dependent. In the worst case, 
					// #categories in each attribute = #rows (e.g., an ID column, say EmployeeID).
					// both inputs are one-dimensional matrices with exact same dimensions, m = size of longer dimension
					worstCaseDim = (mc[0].dimsKnown())
					          ? (mc[0].getRows() > 1 ? mc[0].getRows() : mc[0].getCols() )
							  : (mc[1].getRows() > 1 ? mc[1].getRows() : mc[1].getCols() );
					//note: for ctable histogram dim2 known but automatically replaces m         
					//ret = new long[]{m, m, m};
				}
				
				// Step 2: special handling specified dims
				if( dimsSpec && getInput().get(3) instanceof LiteralOp && getInput().get(4) instanceof LiteralOp ) 
				{
					long outputDim1 = HopRewriteUtils.getIntValueSafe((LiteralOp)getInput().get(3));
					long outputDim2 = HopRewriteUtils.getIntValueSafe((LiteralOp)getInput().get(4));
					long outputNNZ = ( outputDim1*outputDim2 > outputDim1 ? outputDim1 : outputDim1*outputDim2 );
					_dim1 = outputDim1;
					_dim2 = outputDim2;
					return new long[]{outputDim1, outputDim2, outputNNZ};
				}
				
				// Step 3: general case
				//note: for ctable histogram dim2 known but automatically replaces m         
				return new long[]{worstCaseDim, worstCaseDim, worstCaseDim};
			
			case QUANTILE:
				if( mc[2].dimsKnown() )
					return new long[]{mc[2].getRows(), 1, mc[2].getRows()};
				break;
			case IFELSE:
				for(DataCharacteristics lmc : mc)
					if( lmc.dimsKnown() && lmc.getRows() >= 0 ) //known matrix
						return new long[]{lmc.getRows(), lmc.getCols(), -1};
				break;
			case PLUS_MULT:
			case MINUS_MULT:
				//compute back NNz
				double sp1 = OptimizerUtils.getSparsity(mc[0].getRows(), mc[0].getRows(), mc[0].getNonZeros()); 
				double sp2 = OptimizerUtils.getSparsity(mc[2].getRows(), mc[2].getRows(), mc[2].getNonZeros());
				return new long[]{mc[0].getRows(), mc[0].getCols(), (long) Math.min(sp1+sp2,1)};
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
				
		return ret;
	}
	

	@Override
	protected ExecType optFindExecType() 
	{
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) {
			_etype = _etypeForced;
		}
		else
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if ( (getInput().get(0).areDimsBelowThreshold() 
					&& getInput().get(1).areDimsBelowThreshold()
					&& getInput().get(2).areDimsBelowThreshold()) )
				_etype = ExecType.CP;
			else
				_etype = ExecType.SPARK;
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}

		//mark for recompile (forever)
		// additional condition: when execType=CP and additional dimension inputs 
		// are provided (and those values are unknown at initial compile time).
		setRequiresRecompileIfNecessary();
		if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown(true) 
			&& _etype == ExecType.CP && _dimInputsPresent) {
			setRequiresRecompile();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if ( getDataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else 
		{
			switch( _op ) 
			{
				case CTABLE:
					//in general, do nothing because the output size is data dependent
					Hop input1 = getInput().get(0);
					Hop input2 = getInput().get(1);
					Hop input3 = getInput().get(2);
					
					//TODO double check reset (dimsInputPresent?)
					if ( _dim1 == -1 || _dim2 == -1 ) { 
						//for ctable_expand at least one dimension is known
						if( isSequenceRewriteApplicable(true) )
							setDim1( input1._dim1 );
						else if( isSequenceRewriteApplicable(false) )
							setDim2( input2._dim1 );
						//for ctable_histogram also one dimension is known
						Ctable.OperationTypes ternaryOp = Ctable.findCtableOperationByInputDataTypes(
							input1.getDataType(), input2.getDataType(), input3.getDataType());
						if(  ternaryOp==Ctable.OperationTypes.CTABLE_TRANSFORM_HISTOGRAM
							&& input2 instanceof LiteralOp )
						{
							setDim2( HopRewriteUtils.getIntValueSafe((LiteralOp)input2) );
						}
						
						// if output dimensions are provided, update _dim1 and _dim2
						if( getInput().size() >= 5 ) {
							if( getInput().get(3) instanceof LiteralOp )
								setDim1( HopRewriteUtils.getIntValueSafe((LiteralOp)getInput().get(3)) );
							if( getInput().get(4) instanceof LiteralOp )
								setDim2( HopRewriteUtils.getIntValueSafe((LiteralOp)getInput().get(4)) );
						}
					}

					break;
				
				case QUANTILE:
					// This part of the code is executed only when a vector of quantiles are computed
					// Output is a vector of length = #of quantiles to be computed, and it is likely to be dense.
					// TODO qx1
					break;
				
				//default ternary operations
				case IFELSE:
				case PLUS_MULT:
				case MINUS_MULT:
					if( getDataType() == DataType.MATRIX ) {
						setDim1( HopRewriteUtils.getMaxNrowInput(this) );
						setDim2( HopRewriteUtils.getMaxNcolInput(this) );
					}
					break;
				default:
					throw new RuntimeException("Size information for operation (" + _op + ") can not be updated.");
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		TernaryOp ret = new TernaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._dimInputsPresent  = _dimInputsPresent;
		ret._disjointInputs    = _disjointInputs;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof TernaryOp) )
			return false;
		
		TernaryOp that2 = (TernaryOp)that;
		
		//compare basic inputs and weights (always existing)
		boolean ret = (_op == that2._op
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& getInput().get(2) == that2.getInput().get(2));
		
		//compare optional dimension parameters
		ret &= (_dimInputsPresent == that2._dimInputsPresent);
		if( ret && _dimInputsPresent ){
			ret &= getInput().get(3) == that2.getInput().get(3)
				&& getInput().get(4) == that2.getInput().get(4);
		}
		
		//compare optimizer hints and parameters
		ret &= _disjointInputs == that2._disjointInputs
			&& _outputEmptyBlocks == that2._outputEmptyBlocks;
		
		return ret;
	}
	
	private boolean isSequenceRewriteApplicable( boolean left ) 
	{
		boolean ret = false;
		
		//early abort if rewrite globally not allowed
		if( !ALLOW_CTABLE_SEQUENCE_REWRITES )
			return ret;
		
		try
		{
			if( getInput().size()==2 || (getInput().size()==3 && getInput().get(2).getDataType()==DataType.SCALAR) )
			{
				Hop input1 = getInput().get(0);
				Hop input2 = getInput().get(1);
				if( input1.getDataType() == DataType.MATRIX && input2.getDataType() == DataType.MATRIX )
				{
					//probe rewrite on left input
					if( left && input1 instanceof DataGenOp )
					{
						DataGenOp dgop = (DataGenOp) input1;
						if( dgop.getOp() == DataGenMethod.SEQ ){
							Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
							ret = (incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1)
								  || dgop.getIncrementValue()==1.0; //set by recompiler
						}
					}
					//probe rewrite on right input
					if( !left && input2 instanceof DataGenOp )
					{
						DataGenOp dgop = (DataGenOp) input2;
						if( dgop.getOp() == DataGenMethod.SEQ ){
							Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
							ret |= (incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1)
								   || dgop.getIncrementValue()==1.0; //set by recompiler;
						}
					}
				}			
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
			//ret = false;
		}
			
		return ret;
	}
	
	/**
	 * Used for (1) constructing CP lops (hop-lop rewrite), and (2) in order to determine
	 * if dag split after removeEmpty necessary (#2 is precondition for #1). 
	 * 
	 * @return true if ignore zero rewrite
	 */
	public boolean isMatrixIgnoreZeroRewriteApplicable() 
	{
		boolean ret = false;
		
		//early abort if rewrite globally not allowed
		if( !ALLOW_CTABLE_SEQUENCE_REWRITES || _op!=OpOp3.CTABLE )
			return ret;
		
		try
		{
			//1) check for ctable CTABLE_TRANSFORM_SCALAR_WEIGHT
			if( getInput().size()==2 || (getInput().size()>2 && getInput().get(2).getDataType()==DataType.SCALAR) )
			{
				Hop input1 = getInput().get(0);
				Hop input2 = getInput().get(1);
				//2) check for remove empty pair 
				if( input1.getDataType() == DataType.MATRIX && input2.getDataType() == DataType.MATRIX 
					&& input1 instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)input1).getOp()==ParamBuiltinOp.RMEMPTY
					&& input2 instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)input2).getOp()==ParamBuiltinOp.RMEMPTY )
				{
					ParameterizedBuiltinOp pb1 = (ParameterizedBuiltinOp)input1;
					ParameterizedBuiltinOp pb2 = (ParameterizedBuiltinOp)input2;
					Hop pbin1 = pb1.getTargetHop();
					Hop pbin2 = pb2.getTargetHop();
					
					//3) check for reshape pair
					if(    pbin1 instanceof ReorgOp && ((ReorgOp)pbin1).getOp()==ReOrgOp.RESHAPE
						&& pbin2 instanceof ReorgOp && ((ReorgOp)pbin2).getOp()==ReOrgOp.RESHAPE )
					{
						//4) check common non-zero input (this allows to infer two things: 
						//(a) that the dims are equivalent, and zero values for remove empty are aligned)
						Hop left = pbin1.getInput().get(0);
						Hop right = pbin2.getInput().get(0);
						if(    left instanceof BinaryOp && ((BinaryOp)left).getOp()==OpOp2.MULT
							&& left.getInput().get(0) instanceof BinaryOp && ((BinaryOp)left.getInput().get(0)).getOp()==OpOp2.NOTEQUAL
							&& left.getInput().get(0).getInput().get(1) instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left.getInput().get(0).getInput().get(1))==0 
							&& left.getInput().get(0).getInput().get(0) == right ) //relies on CSE
						{	
							ret = true;
						}
						else if(    right instanceof BinaryOp && ((BinaryOp)right).getOp()==OpOp2.MULT
							&& right.getInput().get(0) instanceof BinaryOp && ((BinaryOp)right.getInput().get(0)).getOp()==OpOp2.NOTEQUAL
							&& right.getInput().get(0).getInput().get(1) instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)right.getInput().get(0).getInput().get(1))==0 
							&& right.getInput().get(0).getInput().get(0) == left ) //relies on CSE
						{
							ret = true;
						}
					}
				}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		
		return ret;
	}
}