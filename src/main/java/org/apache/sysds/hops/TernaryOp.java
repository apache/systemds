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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.CentralMoment;
import org.apache.sysds.lops.CoVariance;
import org.apache.sysds.lops.Ctable;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.LopsException;
import org.apache.sysds.lops.PickByCount;
import org.apache.sysds.lops.SortKeys;
import org.apache.sysds.lops.Ternary;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

/** Primary use cases for now, are
 * 		{@code quantile (<n-1-matrix>, <n-1-matrix>, <literal>):      quantile (A, w, 0.5)}
 * 		{@code quantile (<n-1-matrix>, <n-1-matrix>, <scalar>):       quantile (A, w, s)}
 * 		{@code interquantile (<n-1-matrix>, <n-1-matrix>, <scalar>):  interquantile (A, w, s)}
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
public class TernaryOp extends MultiThreadedHop
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
	
	public TernaryOp(String l, DataType dt, ValueType vt, OpOp3 o,
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
	public TernaryOp(String l, DataType dt, ValueType vt, OpOp3 o,
			Hop inp1, Hop inp2, Hop inp3, Hop inp4, Hop inp5, Hop inp6) {
		super(l, dt, vt);
		_op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);
		getInput().add(2, inp3);
		getInput().add(3, inp4);
		getInput().add(4, inp5);
		getInput().add(5, inp6);
		inp1.getParent().add(this);
		inp2.getParent().add(this);
		inp3.getParent().add(this);
		inp4.getParent().add(this);
		inp5.getParent().add(this);
		inp6.getParent().add(this);
		_dimInputsPresent = true;
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
	public boolean isMultiThreadedOpType() {
		return _op == OpOp3.IFELSE
			|| _op == OpOp3.MINUS_MULT
			|| _op == OpOp3.PLUS_MULT
			|| _op == OpOp3.QUANTILE
			|| _op == OpOp3.INTERQUANTILE;
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
				case MAP:
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
	private void constructLopsCentralMoment() {
		if ( _op != OpOp3.MOMENT )
			throw new HopsException("Unexpected operation: " + _op + ", expecting " + OpOp3.MOMENT );
		
		ExecType et = optFindExecType();
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		CentralMoment cm = new CentralMoment(
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(),
			getDataType(), getValueType(), k, et);
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
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		CoVariance cov = new CoVariance(
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(),
			getDataType(), getValueType(), k, et);
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
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		SortKeys sort = SortKeys.constructSortByValueLop(getInput().get(0).constructLops(),
			getInput().get(1).constructLops(), SortKeys.OperationTypes.WithWeights, 
			getInput().get(0).getDataType(), getInput().get(0).getValueType(), et, k);
		PickByCount pick = new PickByCount(sort, getInput().get(2).constructLops(),
			getDataType(), getValueType(), (_op == OpOp3.QUANTILE) ?
			PickByCount.OperationTypes.VALUEPICK : PickByCount.OperationTypes.RANGEPICK, et, true);
		sort.getOutputParameters().setDimensions(getInput().get(0).getDim1(),
			getInput().get(0).getDim2(), getInput().get(0).getBlocksize(), getInput().get(0).getNnz());
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
		boolean outputEmptyBlocks = (getInput().size() == 6) ?
			HopRewriteUtils.getBooleanValue((LiteralOp)getInput(5)) : true;
		
		if( isMatrixIgnoreZeroRewriteApplicable() ) { 
			ignoreZeros = true; //table - rmempty - rshape --> table
			inputLops[0] = ((ParameterizedBuiltinOp)getInput(0)).getTargetHop().getInput(0).constructLops();
			inputLops[1] = ((ParameterizedBuiltinOp)getInput(1)).getTargetHop().getInput(0).constructLops();
		}
		else if( isCTableReshapeRewriteApplicable(et, ternaryOp) ) {
			//table - reshape --> table
			inputLops[0] = ((ReorgOp)getInput(0)).getInput(0).constructLops();
			inputLops[1] = ((ReorgOp)getInput(1)).getInput(0).constructLops();
		}
		
		Ctable ternary = new Ctable(inputLops, ternaryOp,
			getDataType(), getValueType(), ignoreZeros, outputEmptyBlocks, et, OptimizerUtils.getConstrainedNumThreads(getMaxNumThreads()));
		
		ternary.getOutputParameters().setDimensions(getDim1(), getDim2(), getBlocksize(), -1);
		setLineNumbers(ternary);
		
		//force blocked output in CP and SPARK
		ternary.getOutputParameters().setDimensions(getDim1(), getDim2(), getBlocksize(), -1);
		
		//ternary opt, w/o reblock in CP
		setLops(ternary);
	}

	private void constructLopsTernaryDefault() {
		ExecType et = optFindExecType();
		int k = 1;
		if( getInput().stream().allMatch(h -> h.getDataType().isScalar()) )
			et = ExecType.CP; //always CP for pure scalar operations
		else
			k= OptimizerUtils.getConstrainedNumThreads( _maxNumThreads );
		
		Ternary plusmult = new Ternary(_op,
			getInput(0).constructLops(),
			getInput(1).constructLops(),
			getInput(2).constructLops(), 
			getDataType(),getValueType(), et, k );
		setOutputDimensions(plusmult);
		setLineNumbers(plusmult);
		setLops(plusmult);
		
		if( _op==OpOp3.IFELSE && HopRewriteUtils.isData(getInput(0), OpOpData.TRANSIENTREAD, DataType.SCALAR))
			setRequiresRecompile(); //good chance of removing ops via literal replacements + rewrites
	}
	
	@Override
	public String getOpString() {
		return "t(" + _op.toString() + ")";
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
			case MAP:
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
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz ) {
		double ret = 0;
		if( _op == OpOp3.CTABLE ) {
			if ( dimsKnown() ) {
				// output dimensions are known, and hence a MatrixBlock is allocated
				double sp = OptimizerUtils.getSparsity(getDim1(), getDim2(), Math.min(nnz, getDim1()));
				ret = OptimizerUtils.estimateSizeExactSparsity(getDim1(), getDim2(), sp );
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
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		DataCharacteristics[] mc = memo.getAllInputStats(getInput());
		DataCharacteristics ret = null;
		
		switch( _op ) 
		{
			case MAP:
				long ldim1 = (mc[0].rowsKnown()) ? mc[0].getRows() :
					(mc[1].getRows()>=1) ? mc[1].getRows() : -1;
				long ldim2 = (mc[0].colsKnown()) ? mc[0].getCols() :
					(mc[1].getCols()>=1) ? mc[1].getCols() : -1;
				if( ldim1>=0 && ldim2>=0 )
					ret = new MatrixCharacteristics(ldim1, ldim2, -1, (long) (ldim1 * ldim2 * 1.0));
				return ret;

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
					setDim1(outputDim1);
					setDim2(outputDim2);
					return new MatrixCharacteristics(outputDim1, outputDim2, -1, outputNNZ);
				}
				
				// Step 3: general case
				//note: for ctable histogram dim2 known but automatically replaces m
				return new MatrixCharacteristics(worstCaseDim, worstCaseDim, -1, worstCaseDim);
			
			case QUANTILE:
				if( mc[2].dimsKnown() )
					return new MatrixCharacteristics(mc[2].getRows(), 1, -1, mc[2].getRows());
				break;
			case IFELSE:
				for(DataCharacteristics lmc : mc)
					if( lmc.dimsKnown() && lmc.getRows() >= 0 ) //known matrix
						return new MatrixCharacteristics(lmc.getRows(), lmc.getCols(), -1, -1);
				break;
			case PLUS_MULT:
			case MINUS_MULT:
				//compute back NNz
				double sp1 = OptimizerUtils.getSparsity(mc[0].getRows(), mc[0].getRows(), mc[0].getNonZeros()); 
				double sp2 = OptimizerUtils.getSparsity(mc[2].getRows(), mc[2].getRows(), mc[2].getNonZeros());
				return new MatrixCharacteristics(mc[0].getRows(), mc[0].getCols(), -1, (long) Math.min(sp1+sp2,1));
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
		
		return ret;
	}
	

	public ExecType findExecTypeTernaryOp(){
		return _etype == null ? optFindExecType(OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE) : _etype;
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) 
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
		if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown() 
			&& _etype == ExecType.CP && _dimInputsPresent) {
			setRequiresRecompile();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);
		Hop input3 = getInput().get(2);

		if ( getDataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else
		{
			switch( _op ) 
			{
				case MAP:
					long ldim1, ldim2, lnnz1 = -1;
					ldim1 = (input1.rowsKnown()) ? input1.getDim1() : ((input2.getDim1()>=1)?input2.getDim1():-1);
					ldim2 = (input1.colsKnown()) ? input1.getDim2() : ((input2.getDim2()>=1)?input2.getDim2():-1);
					lnnz1 = input1.getNnz();

					setDim1( ldim1 );
					setDim2( ldim2 );
					setNnz(lnnz1);
					break;
				case CTABLE:
					//in general, do nothing because the output size is data dependent

					//TODO double check reset (dimsInputPresent?)
					if ( !dimsKnown() ) { 
						//for ctable_expand at least one dimension is known
						if( isSequenceRewriteApplicable(true) )
							setDim1( input1.getDim1() );
						else if( isSequenceRewriteApplicable(false) )
							setDim2( input2.getDim1() );
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
							refreshRowsParameterInformation(getInput(3));
							refreshColsParameterInformation(getInput(4));
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

		if(_op == OpOp3.MAP)
			return false; // custom UDFs
		
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
	
	public boolean isSequenceRewriteApplicable( boolean left ) 
	{
		boolean ret = false;
		
		//early abort if rewrite globally not allowed
		if( !ALLOW_CTABLE_SEQUENCE_REWRITES )
			return ret;
		
		try
		{
			// TODO: to rewrite is not currently not triggered if outdim are given --> getInput().size()>=3
			// currently disabled due performance decrease
			if( getInput().size()==2 || (getInput().size()==3 && getInput().get(2).getDataType()==DataType.SCALAR) )
			{
				Hop input1 = getInput().get(0);
				Hop input2 = getInput().get(1);
				if( (input1.getDataType() == DataType.MATRIX 
					|| input1.getDataType() == DataType.SCALAR )
					 && input2.getDataType() == DataType.MATRIX )
				{
					//probe rewrite on left input
					if( left && input1 instanceof DataGenOp )
					{
						DataGenOp dgop = (DataGenOp) input1;
						if( dgop.getOp() == OpOpDG.SEQ ){
							Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
							ret = (incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1)
								  || dgop.getIncrementValue()==1.0; //set by recompiler
						}
					}
					if( left && input1 instanceof LiteralOp && ((LiteralOp)input1).getStringValue().contains("seq(")){
						ret = true;
					}
					//probe rewrite on right input
					if( !left && input2 instanceof DataGenOp )
					{
						DataGenOp dgop = (DataGenOp) input2;
						if( dgop.getOp() == OpOpDG.SEQ ){
							Hop incr = dgop.getInput().get(dgop.getParamIndex(Statement.SEQ_INCR));
							ret |= (incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1)
								   || dgop.getIncrementValue()==1.0; //set by recompiler;
						}
					}
				}
			}
		}
		catch(Exception ex) {
			throw new HopsException(ex);
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
	
	public boolean isCTableReshapeRewriteApplicable(ExecType et, Ctable.OperationTypes opType) {
		//early abort if rewrite globally not allowed
		if( !ALLOW_CTABLE_SEQUENCE_REWRITES || _op!=OpOp3.CTABLE || (et!=ExecType.CP && et!=ExecType.SPARK) )
			return false;
		
		//1) check for ctable CTABLE_TRANSFORM_SCALAR_WEIGHT
		if( opType==Ctable.OperationTypes.CTABLE_TRANSFORM_SCALAR_WEIGHT ) {
			Hop input1 = getInput().get(0);
			Hop input2 = getInput().get(1);
			//2) check for reshape pair
			if(    input1 instanceof ReorgOp && ((ReorgOp)input1).getOp()==ReOrgOp.RESHAPE
				&& input2 instanceof ReorgOp && ((ReorgOp)input2).getOp()==ReOrgOp.RESHAPE )
			{
				//common byrow parameter
				return input1.getInput(4) == input2.getInput(4) //CSE
					|| input1.getInput(4).compare(input2.getInput(4));
			}
		}
		
		return false;
	}
}
