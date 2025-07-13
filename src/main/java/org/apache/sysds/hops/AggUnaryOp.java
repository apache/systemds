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
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.PartialAggregate;
import org.apache.sysds.lops.TernaryAggregate;
import org.apache.sysds.lops.UAggOuterChain;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.List;

// Aggregate unary (cell) operation: Sum (aij), col_sum, row_sum

public class AggUnaryOp extends MultiThreadedHop
{
	private static final boolean ALLOW_UNARYAGG_WO_FINAL_AGG = true;
	
	private AggOp _op;
	private Direction _direction;

	private AggUnaryOp() {
		//default constructor for clone
	}
	
	public AggUnaryOp(String l, DataType dt, ValueType vt, AggOp o, Direction idx, Hop inp) 
	{
		super(l, dt, vt);
		_op = o;
		_direction = idx;
		getInput().add(0, inp);
		inp.getParent().add(this);
	}

	public AggOp getOp() {
		return _op;
	}
	
	public void setOp(AggOp op) {
		_op = op;
	}
	
	public Direction getDirection() {
		return _direction;
	}
	
	public void setDirection(Direction direction) {
		_direction = direction;
	}

	
	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		
		try {
			if( isTernaryAggregateRewriteApplicable() || isUnaryAggregateOuterCPRewriteApplicable() ) {
				return false;
			}
			else if ((_op == AggOp.SUM    && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.SUM_SQ && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.MAX    && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.MIN    && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.MEAN   && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.VAR    && (_direction == Direction.RowCol || _direction == Direction.Row || _direction == Direction.Col))
					 || (_op == AggOp.PROD   && (_direction == Direction.RowCol))){
				return true;
			}
		} catch (HopsException e) {
			throw new RuntimeException(e);
		}
		return false;
	}
	
	@Override
	public Lop constructLops()
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		try 
		{
			ExecType et = optFindExecType();
			Hop input = getInput().get(0);
			
			if ( et == ExecType.CP || et == ExecType.GPU || et == ExecType.FED || et == ExecType.OOC  )
			{
				Lop agg1 = null; 
				if( isTernaryAggregateRewriteApplicable() ) {
					agg1 = constructLopsTernaryAggregateRewrite(et);
				}
				else if( et != ExecType.FED && isUnaryAggregateOuterCPRewriteApplicable() )
				{
					BinaryOp binput = (BinaryOp)getInput().get(0);
					agg1 = new UAggOuterChain( binput.getInput().get(0).constructLops(), 
							binput.getInput().get(1).constructLops(), _op, _direction, 
							binput.getOp(), DataType.MATRIX, getValueType(), ExecType.CP);
					PartialAggregate.setDimensionsBasedOnDirection(agg1, getDim1(), getDim2(), input.getBlocksize(), _direction);
				
					if (getDataType() == DataType.SCALAR) {
						UnaryCP unary1 = new UnaryCP(agg1, OpOp1.CAST_AS_SCALAR,
							getDataType(), getValueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, -1);
						setLineNumbers(unary1);
						agg1 = unary1;
					}
				
				}
				else { //general case
					int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
					agg1 = new PartialAggregate(input.constructLops(),
							_op, _direction, getDataType(),getValueType(), et, k);
				}

				setOutputDimensions(agg1);
				setLineNumbers(agg1);
				setLops(agg1);
				
				if (getDataType() == DataType.SCALAR)
					agg1.getOutputParameters().setDimensions(1, 1, getBlocksize(), getNnz());
			}
			else if( et == ExecType.SPARK )
			{
				//unary aggregate
				if( isTernaryAggregateRewriteApplicable() ) 
				{
					Lop aggregate = constructLopsTernaryAggregateRewrite(et);
					setOutputDimensions(aggregate); //0x0 (scalar)
					setLineNumbers(aggregate);
					setLops(aggregate);
				}
				else if( isUnaryAggregateOuterSPRewriteApplicable() ) 
				{
					BinaryOp binput = (BinaryOp)getInput().get(0);
					Lop transform1 = new UAggOuterChain( binput.getInput().get(0).constructLops(), 
							binput.getInput().get(1).constructLops(), _op, _direction, 
							binput.getOp(), DataType.MATRIX, getValueType(), ExecType.SPARK);
					PartialAggregate.setDimensionsBasedOnDirection(transform1, getDim1(), getDim2(), input.getBlocksize(), _direction);
					setLineNumbers(transform1);
					setLops(transform1);
				
					if (getDataType() == DataType.SCALAR) {
						UnaryCP unary1 = new UnaryCP(transform1,
							OpOp1.CAST_AS_SCALAR, getDataType(), getValueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, -1);
						setLineNumbers(unary1);
						setLops(unary1);
					}
				
				}
				else //default
				{
					boolean needAgg = requiresAggregation(input, _direction);
					SparkAggType aggtype = getSparkUnaryAggregationType(needAgg);
					
					PartialAggregate aggregate = new PartialAggregate(input.constructLops(), 
						_op, _direction, input._dataType, getValueType(), aggtype, et);
					aggregate.setDimensionsBasedOnDirection(getDim1(), getDim2(), input.getBlocksize());
					setLineNumbers(aggregate);
					setLops(aggregate);
				
					if (getDataType() == DataType.SCALAR) {
						UnaryCP unary1 = new UnaryCP(aggregate, 
							OpOp1.CAST_AS_SCALAR, getDataType(), getValueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, -1);
						setLineNumbers(unary1);
						setLops(unary1);
					}
				}
			}
			else throw new HopsException("ExecType " + et + " not recognized in " + this.toString() );
		} 
		catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "In AggUnary Hop, error constructing Lops " , e);
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		//return created lops
		return getLops();
	}

	
	
	@Override
	public String getOpString() {
		//ua - unary aggregate, for consistency with runtime
		return"ua(" + _op.toString() + _direction.toString() + ")";
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		double sparsity = -1;
		if (isGPUEnabled()) {
			// The GPU version (for the time being) only does dense outputs
			sparsity = 1.0;
		} else {
			sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		}

		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		 //default: no additional memory required
		double val = 0;
		
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		
		switch( _op ) //see MatrixAggLib for runtime operations
		{
			case MAX:
			case MIN:
				//worst-case: column-wise, sparse (temp int count arrays)
				if( _direction == Direction.Col )
					val = dim2 * OptimizerUtils.INT_SIZE;
				break;
			case SUM:
			case SUM_SQ:
				//worst-case correction LASTROW / LASTCOLUMN
				if( _direction == Direction.Col ) //(potentially sparse)
					val = OptimizerUtils.estimateSizeExactSparsity(2, dim2, sparsity);
				else if( _direction == Direction.Row ) //(always dense)
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 2, 1.0);
				break;
			case MEAN:
				//worst-case correction LASTTWOROWS / LASTTWOCOLUMNS
				if( _direction == Direction.Col ) //(potentially sparse)
					val = OptimizerUtils.estimateSizeExactSparsity(3, dim2, sparsity);
				else if( _direction == Direction.Row ) //(always dense)
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 3, 1.0);
				break;
			case VAR:
				//worst-case correction LASTFOURROWS / LASTFOURCOLUMNS
				if (isGPUEnabled()) {
					// The GPU implementation only operates on dense data
					// It allocates 2 dense blocks to help with these ops:
					// Assume Y = var(X) Or colVars(X), Or rowVars(X)
					// 1. Y = mean/rowMeans/colMeans(X)               <-- Y is a scalar or row-vector or col-vector
					// 2. temp1 = X - Y                               <-- temp1 is a matrix of size(X)
					// 3. temp2 = temp1 ^ 2                           <-- temp2 is a matrix of size(X)
					// 4. temp3 = sum/rowSums/colSums(temp2)          <-- temp3 is a scalar or a row-vector or col-vector
					// 5. Y = temp3 / (size(X) or nrow(X) or ncol(X)) <-- Y is a scalar or a row-vector or col-vector

					long in1dim1 = getInput().get(0).getDim1();
					long in1dim2 = getInput().get(0).getDim2();

					val = 2 * OptimizerUtils.estimateSize(in1dim1, in1dim2);    // For temp1 & temp2
					if (_direction == Direction.Col){
						val += OptimizerUtils.estimateSize(in1dim1, 1);   // For temp3
					} else if (_direction == Direction.Row){
						val += OptimizerUtils.estimateSize(1, in1dim2);  // For temp3
					}

				} else if( _direction == Direction.Col ) { //(potentially sparse)
					val = OptimizerUtils.estimateSizeExactSparsity(5, dim2, sparsity);
				} else if( _direction == Direction.Row ) { //(always dense)
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 5, 1.0);
				}
				break;
			case MAXINDEX:
			case MININDEX:
				Hop hop = getInput().get(0);
				if(isUnaryAggregateOuterCPRewriteApplicable())
					val = 3 * OptimizerUtils.estimateSizeExactSparsity(1, hop.getDim2(), 1.0);
				else
					//worst-case correction LASTCOLUMN 
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 2, 1.0);
				break;
			default:
				//no intermediate memory consumption
				val = 0;
		}
		
		return val;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo ) {
		DataCharacteristics ret = null;
		Hop input = getInput().get(0);
		DataCharacteristics dc = memo.getAllInputStats(input);
		if( _op == AggOp.UNIQUE ) {
			if( _direction == Direction.RowCol && dc.rowsKnown() )
				ret = new MatrixCharacteristics(dc.getRows(), 1, -1, -1);
			else
				ret = new MatrixCharacteristics(dc.getRows(), dc.getCols(), -1, -1);
		}
		else {
			if( _direction == Direction.Col && dc.colsKnown() )
				ret = new MatrixCharacteristics(1, dc.getCols(), -1, -1);
			else if( _direction == Direction.Row && dc.rowsKnown() )
				ret = new MatrixCharacteristics(dc.getRows(), 1, -1, -1);
		}
		return ret;
	}
	

	private boolean inputAlreadySpark(){
		return (!(getInput(0) instanceof DataOp)  //input is not checkpoint
		&& getInput(0).optFindExecType() == ExecType.SPARK);
	}

	private boolean inputOnlyRDD(){
		return (getInput(0) instanceof DataOp && ((DataOp)getInput(0)).hasOnlyRDD());
	} 

	private boolean onlyOneParent(){
		return getInput(0).getParent().size()==1;
	}

	private boolean allParentsSpark(){
		return getInput(0).getParent().stream().filter(h -> h != this)
					.allMatch(h -> h.optFindExecType(false) == ExecType.SPARK);
	}

	private boolean inputDoesNotRequireAggregation(){
		return !requiresAggregation(getInput(0), _direction);
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = ExecType.SPARK;
		
		//forced / memory-based / threshold-based decision
		if( _etypeForced != null ) {
			setExecType(_etypeForced);
		}
		else
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel()) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if(getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector()) {
				setExecType(ExecType.CP);
			}
			else {
				setExecType(REMOTE);
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}

		//spark-specific decision refinement (execute unary aggregate w/ spark input and 
		//single parent also in spark because it's likely cheap and reduces data transfer)
		//we also allow multiple parents, if all other parents are already in Spark mode

		boolean shouldEvaluateIfSpark =  transitive && _etype == ExecType.CP && _etypeForced != ExecType.CP;

		if( shouldEvaluateIfSpark
			&& (inputAlreadySpark() || inputOnlyRDD())
			&& (onlyOneParent() || allParentsSpark() || inputDoesNotRequireAggregation() ))
		{
			//pull unary aggregate into spark 
			setExecType(ExecType.SPARK);
		}

		//ensure cp exec type for single-node operations
		if( _op == AggOp.UNIQUE )
			setExecType(ExecType.CP);
		else
			setRequiresRecompileIfNecessary();
		
		return _etype;
	}

	private static boolean requiresAggregation( Hop input, Direction dir ) 
	{
		if( !ALLOW_UNARYAGG_WO_FINAL_AGG )
			return false; //customization not allowed
		
		boolean noAggRequired = 
				  ( input.getDim1()>1 && input.getDim1()<=input.getBlocksize() && dir==Direction.Col ) //e.g., colSums(X) with nrow(X)<=1000
				||( input.getDim2()>1 && input.getDim2()<=input.getBlocksize() && dir==Direction.Row ); //e.g., rowSums(X) with ncol(X)<=1000
	
		return !noAggRequired;
	}

	private SparkAggType getSparkUnaryAggregationType( boolean agg ) {
		if( !agg )
			return SparkAggType.NONE;
		if(   getDataType()==DataType.SCALAR //in case of scalars the block dims are not set
		   || dimsKnown() && getDim1()<=getBlocksize() && getDim2()<=getBlocksize() )
			return SparkAggType.SINGLE_BLOCK;
		return SparkAggType.MULTI_BLOCK;
	}

	private boolean isTernaryAggregateRewriteApplicable() 
	{
		boolean ret = false;
		
		// TODO: Disable ternary aggregate rewrite on GPU backend.
		if(DMLScript.USE_ACCELERATOR)
			return false;
		
		//currently we support only sum over binary multiply but potentially 
		//it can be generalized to any RC aggregate over two common binary operations
		if( OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES && _op == AggOp.SUM &&
			(_direction == Direction.RowCol || _direction == Direction.Col)  ) 
		{
			Hop input1 = getInput().get(0);
			if (input1.getParent().size() == 1
					&& input1 instanceof BinaryOp) { //sum single consumer
				BinaryOp binput1 = (BinaryOp)input1;

				if (binput1.getOp() == OpOp2.POW
						&& binput1.getInput().get(1) instanceof LiteralOp) {
					LiteralOp lit = (LiteralOp)binput1.getInput().get(1);
					ret = HopRewriteUtils.getIntValueSafe(lit) == 3;
				}
				else if (binput1.getOp() == OpOp2.MULT ) {
					Hop input11 = input1.getInput().get(0);
					Hop input12 = input1.getInput().get(1);

					if (input11 instanceof BinaryOp && ((BinaryOp) input11).getOp() == OpOp2.MULT) {
						//ternary, arbitrary matrices but no mv/outer operations.
						ret = HopRewriteUtils.isEqualSize(input11.getInput().get(0), input1) && HopRewriteUtils
								.isEqualSize(input11.getInput().get(1), input1) && HopRewriteUtils
								.isEqualSize(input12, input1);
					} else if (input12 instanceof BinaryOp && ((BinaryOp) input12).getOp() == OpOp2.MULT) {
						//ternary, arbitrary matrices but no mv/outer operations.
						ret = HopRewriteUtils.isEqualSize(input12.getInput().get(0), input1) && HopRewriteUtils
								.isEqualSize(input12.getInput().get(1), input1) && HopRewriteUtils
								.isEqualSize(input11, input1);
					} else {
						//binary, arbitrary matrices but no mv/outer operations.
						ret = HopRewriteUtils.isEqualSize(input11, input12);
					}
				}
			}
			if (input1.getParent().size() == 1
					&& input1 instanceof NaryOp) { //sum single consumer
				NaryOp nop = (NaryOp) input1;
				if(nop.getOp() == Types.OpOpN.MULT){
					List<Hop> inputsN = nop.getInput();
					if(inputsN.size() == 3){
						ret = HopRewriteUtils.isEqualSize(inputsN.get(0), inputsN.get(1)) &&
								HopRewriteUtils.isEqualSize(inputsN.get(1), inputsN.get(2));
					}
				}
			}
		}
		return ret;
	}
	
	private static boolean isCompareOperator(OpOp2 opOp2)
	{
		return (opOp2 == OpOp2.LESS || opOp2 == OpOp2.LESSEQUAL 
			|| opOp2 == OpOp2.GREATER || opOp2 == OpOp2.GREATEREQUAL
			|| opOp2 == OpOp2.EQUAL || opOp2 == OpOp2.NOTEQUAL);
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return true;
	}
	
	/**
	 * This will check if there is sufficient memory locally (twice the size of second matrix, for original and sort data), and remotely (size of second matrix (sorted data)).  
	 * @return true if sufficient memory
	 */
	private boolean isUnaryAggregateOuterSPRewriteApplicable() 
	{
		boolean ret = false;
		Hop input = getInput().get(0);
		
		if( input instanceof BinaryOp && ((BinaryOp)input).isOuter() )
		{
			//note: both cases (partitioned matrix, and sorted double array), require to
			//fit the broadcast twice into the local memory budget. Also, the memory 
			//constraint only needs to take the rhs into account because the output is 
			//guaranteed to be an aggregate of <=16KB
			
			Hop right = input.getInput().get(1);
			
			double size = right.dimsKnown() ? 
					OptimizerUtils.estimateSize(right.getDim1(), right.getDim2()) : //dims known and estimate fits
					right.getOutputMemEstimate();                      //dims unknown but worst-case estimate fits
			
			if(_op == AggOp.MAXINDEX || _op == AggOp.MININDEX){
				double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
				double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

				//basic requirement: the broadcast needs to to fit twice in the remote broadcast memory 
				//and local memory budget because we have to create a partitioned broadcast
				//memory and hand it over to the spark context as in-memory object
				ret = ( 2*size < memBudgetExec && 2*size < memBudgetLocal );
			
			} else {
				if( OptimizerUtils.checkSparkBroadcastMemoryBudget(size) ) {
					ret = true;
				}
			}
				
		}
		
		return ret;
	}
	
	
	
	/**
	 * This will check if this is one of the operator from supported LibMatrixOuterAgg library.
	 * It needs to be Outer, aggregator type SUM, RowIndexMin, RowIndexMax and 6 operators <, <=, >, >=, == and !=
	 *   
	 *   
	 * @return true if unary aggregate outer
	 */
	private boolean isUnaryAggregateOuterCPRewriteApplicable() {
		boolean ret = false;
		Hop input = getInput().get(0);
		if(( input instanceof BinaryOp && ((BinaryOp)input).isOuter() )
			&& (_op == AggOp.MAXINDEX || _op == AggOp.MININDEX || _op == AggOp.SUM)
			&& (isCompareOperator(((BinaryOp)input).getOp())))
			ret = true;
		return ret;
	}

	private Lop constructLopsTernaryAggregateRewrite(ExecType et) 
	{
		Lop in1 = null, in2 = null, in3 = null;
		Hop input = getInput().get(0);
		if(input instanceof BinaryOp) {
			BinaryOp input1 = (BinaryOp) input;
			Hop input11 = input1.getInput().get(0);
			Hop input12 = input1.getInput().get(1);

			boolean handled = false;

			if (input1.getOp() == OpOp2.POW) {
				assert (HopRewriteUtils.isLiteralOfValue(input12, 3)) : "this case can only occur with a power of 3";
				in1 = input11.constructLops();
				in2 = in1;
				in3 = in1;
				handled = true;
			} else if (HopRewriteUtils.isBinary(input11, OpOp2.MULT, OpOp2.POW)) {
				BinaryOp b11 = (BinaryOp) input11;
				switch (b11.getOp()) {
					case MULT: // A*B*C case
						in1 = input11.getInput().get(0).constructLops();
						in2 = input11.getInput().get(1).constructLops();
						in3 = input12.constructLops();
						handled = true;
						break;
					case POW: // A*A*B case
						Hop b112 = b11.getInput().get(1);
						if (!(input12 instanceof BinaryOp && ((BinaryOp) input12).getOp() == OpOp2.MULT)
								&& HopRewriteUtils.isLiteralOfValue(b112, 2)) {
							in1 = b11.getInput().get(0).constructLops();
							in2 = in1;
							in3 = input12.constructLops();
							handled = true;
						}
						break;
					default:
						break;
				}
			} else if (HopRewriteUtils.isBinary(input12, OpOp2.MULT, OpOp2.POW)) {
				BinaryOp b12 = (BinaryOp) input12;
				switch (b12.getOp()) {
					case MULT: // A*B*C case
						in1 = input11.constructLops();
						in2 = input12.getInput().get(0).constructLops();
						in3 = input12.getInput().get(1).constructLops();
						handled = true;
						break;
					case POW: // A*B*B case
						Hop b112 = b12.getInput().get(1);
						if (HopRewriteUtils.isLiteralOfValue(b112, 2)) {
							in1 = b12.getInput().get(0).constructLops();
							in2 = in1;
							in3 = input11.constructLops();
							handled = true;
						}
						break;
					default:
						break;
				}
			}

			if (!handled) {
				in1 = input11.constructLops();
				in2 = input12.constructLops();
				in3 = new LiteralOp(1).constructLops();
			}
		} else {
			NaryOp input1 = (NaryOp) input;
			in1 = input1.getInput().get(0).constructLops();
			in2 = input1.getInput().get(1).constructLops();
			in3 = input1.getInput().get(2).constructLops();
		}

		//create new ternary aggregate operator
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		// The execution type of a unary aggregate instruction should depend on the execution type of inputs to avoid OOM
		// Since we only support matrix-vector and not vector-matrix, checking the execution type of input1 should suffice.
		ExecType et_input = input.optFindExecType();
		// Because ternary aggregate are not supported on GPU
		et_input = et_input == ExecType.GPU ? ExecType.CP : et_input;
		// If forced ExecType is FED, it means that the federated planner updated the ExecType and
		// execution may fail if ExecType is not FED
		et_input = (getForcedExecType() == ExecType.FED) ? ExecType.FED : et_input;

		return new TernaryAggregate(in1, in2, in3, AggOp.SUM,
				OpOp2.MULT, _direction, getDataType(), ValueType.FP64, et_input, k);
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input = getInput().get(0);
		if( _op == AggOp.UNIQUE ) {
			if ( _direction == Direction.Col ) {
				setDim1(-1); //unknown num unique
				setDim2(input.getDim2());
			}
			else if ( _direction == Direction.Row ) {
				setDim1(input.getDim1());
				setDim2(-1); //unknown num unique
			}
			else {
				setDim1(-1);
				setDim2(1);
			}
		}
		//general case: all other unary aggregations 
		else if (getDataType() != DataType.SCALAR) {
			if ( _direction == Direction.Col ) //colwise computations
			{
				setDim1(1);
				setDim2(input.getDim2());
			}
			else if ( _direction == Direction.Row )
			{
				setDim1(input.getDim1());
				setDim2(1);	
			}
		}
	}
	
	@Override
	public boolean isTransposeSafe()
	{
		boolean ret = (_direction == Direction.RowCol) && //full aggregate
		              (_op == AggOp.SUM || _op == AggOp.SUM_SQ || //valid aggregration functions
		               _op == AggOp.MIN || _op == AggOp.MAX ||
		               _op == AggOp.PROD || _op == AggOp.MEAN ||
		               _op == AggOp.VAR);
		//note: trace and maxindex are not transpose-safe.
		
		return ret;	
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		AggUnaryOp ret = new AggUnaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._direction = _direction;
		ret._maxNumThreads = _maxNumThreads;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof AggUnaryOp) )
			return false;
		
		AggUnaryOp that2 = (AggUnaryOp)that;
		return (   _op == that2._op
				&& _direction == that2._direction
				&& _maxNumThreads == that2._maxNumThreads
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
