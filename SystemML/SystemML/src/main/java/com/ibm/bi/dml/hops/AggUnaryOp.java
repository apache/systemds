/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.AggBinaryOp.SparkAggType;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Aggregate.OperationTypes;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes;
import com.ibm.bi.dml.lops.TernaryAggregate;
import com.ibm.bi.dml.lops.UAggOuterChain;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;


/* Aggregate unary (cell) operation: Sum (aij), col_sum, row_sum
 * 		Properties: 
 * 			Symbol: +, min, max, ...
 * 			1 Operand
 * 	
 * 		Semantic: generate indices, align, aggregate
 */

public class AggUnaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	
	public AggOp getOp()
	{
		return _op;
	}
	
	public void setOp(AggOp op)
	{
		_op = op;
	}
	
	public Direction getDirection()
	{
		return _direction;
	}
	
	public void setDirection(Direction direction)
	{
		_direction = direction;
	}

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{	
		//return already created lops
		if( getLops() != null )
			return getLops();

		try 
		{
			ExecType et = optFindExecType();
			Hop input = getInput().get(0);
			
			if ( et == ExecType.CP ) 
			{
				Lop agg1 = null;
				if( isTernaryAggregateRewriteApplicable() ) {
					agg1 = constructLopsTernaryAggregateRewrite(et);
				}
				else { //general case
					agg1 = new PartialAggregate(input.constructLops(), 
							HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), getDataType(),getValueType(), et);
				}
				
				setOutputDimensions(agg1);
				setLineNumbers(agg1);
				setLops(agg1);
				
				if (getDataType() == DataType.SCALAR) {
					agg1.getOutputParameters().setDimensions(1, 1, getRowsInBlock(), getColsInBlock(), getNnz());
				}
			}
			else if( et == ExecType.MR )
			{
				OperationTypes op = HopsAgg2Lops.get(_op);
				DirectionTypes dir = HopsDirection2Lops.get(_direction);
				
				//unary aggregate operation
				Lop transform1 = null;
				if( isUnaryAggregateOuterRewriteApplicable() ) 
				{
					BinaryOp binput = (BinaryOp)getInput().get(0);
					transform1 = new UAggOuterChain( binput.getInput().get(0).constructLops(), 
							binput.getInput().get(1).constructLops(), op, dir, 
							HopsOpOp2LopsB.get(binput.getOp()), DataType.MATRIX, getValueType(), ExecType.MR);
					PartialAggregate.setDimensionsBasedOnDirection(transform1, getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock(), dir);
				}
				else //default
				{
					transform1 = new PartialAggregate(input.constructLops(), op, dir, DataType.MATRIX, getValueType());
					((PartialAggregate) transform1).setDimensionsBasedOnDirection(getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock());
				}
				setLineNumbers(transform1);
				
				//aggregation if required
				Lop aggregate = null;
				Group group1 = null; 
				Aggregate agg1 = null;
				if( requiresAggregation(input, _direction) || transform1 instanceof UAggOuterChain )
				{
					group1 = new Group(transform1, Group.OperationTypes.Sort, DataType.MATRIX, getValueType());
					group1.getOutputParameters().setDimensions(getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock(), getNnz());
					setLineNumbers(group1);
					
					agg1 = new Aggregate(group1, HopsAgg2Lops.get(_op), DataType.MATRIX, getValueType(), et);
					agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock(), getNnz());
					agg1.setupCorrectionLocation(PartialAggregate.getCorrectionLocation(op,dir));
					setLineNumbers(agg1);
					
					aggregate = agg1;
				}
				else
				{
					((PartialAggregate) transform1).setDropCorrection();
					aggregate = transform1;
				}
				
				setLops(aggregate);

				//cast if required
				if (getDataType() == DataType.SCALAR) {

					// Set the dimensions of PartialAggregate LOP based on the
					// direction in which aggregation is performed
					PartialAggregate.setDimensionsBasedOnDirection(transform1, input.getDim1(), input.getDim2(),
							input.getRowsInBlock(), input.getColsInBlock(), dir);
					
					if( group1 != null && agg1 != null ) { //if aggregation required
						group1.getOutputParameters().setDimensions(input.getDim1(), input.getDim2(), 
								input.getRowsInBlock(), input.getColsInBlock(), getNnz());
						agg1.getOutputParameters().setDimensions(1, 1, 
								input.getRowsInBlock(), input.getColsInBlock(), getNnz());
					}
					
					UnaryCP unary1 = new UnaryCP(
							aggregate, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
							getDataType(), getValueType());
					unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					setLineNumbers(unary1);
					setLops(unary1);
				}
			}
			else if( et == ExecType.SPARK )
			{
				OperationTypes op = HopsAgg2Lops.get(_op);
				DirectionTypes dir = HopsDirection2Lops.get(_direction);

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
							binput.getInput().get(1).constructLops(), op, dir, 
							HopsOpOp2LopsB.get(binput.getOp()), DataType.MATRIX, getValueType(), ExecType.SPARK);
					PartialAggregate.setDimensionsBasedOnDirection(transform1, getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock(), dir);
					setLineNumbers(transform1);
					setLops(transform1);
				}				
				else //default
				{
					boolean needAgg = requiresAggregation(input, _direction);
					SparkAggType aggtype = getSparkUnaryAggregationType(needAgg);
					
					PartialAggregate aggregate = new PartialAggregate(input.constructLops(), 
							HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), DataType.MATRIX, getValueType(), aggtype, et);
					aggregate.setDimensionsBasedOnDirection(getDim1(), getDim2(), input.getRowsInBlock(), input.getColsInBlock());
					setLineNumbers(aggregate);
					setLops(aggregate);
				
					if (getDataType() == DataType.SCALAR) {
						UnaryCP unary1 = new UnaryCP(aggregate, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
								                    getDataType(), getValueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
						setLineNumbers(unary1);
						setLops(unary1);
					}
				}
			}
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
		String s = "ua(" + 
				HopsAgg2String.get(_op) + 
				HopsDirection2String.get(_direction) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				LOG.debug("  Direction: " + _direction);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
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
				//worst-case correction LASTROW / LASTCOLUMN 
				if( _direction == Direction.Col ) //(potentially sparse)
					val = OptimizerUtils.estimateSizeExactSparsity(1, dim2, sparsity);
				else if( _direction == Direction.Row ) //(always dense)
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 1, 1.0);
				break;
			case MEAN:
				//worst-case correction LASTTWOROWS / LASTTWOCOLUMNS 
				if( _direction == Direction.Col ) //(potentially sparse)
					val = OptimizerUtils.estimateSizeExactSparsity(2, dim2, sparsity);
				else if( _direction == Direction.Row ) //(always dense)
					val = OptimizerUtils.estimateSizeExactSparsity(dim1, 2, 1.0);
				break;
			case MAXINDEX:
			case MININDEX:
				//worst-case correction LASTCOLUMN 
				val = OptimizerUtils.estimateSizeExactSparsity(dim1, 1, 1.0);
				break;
			default:
				//no intermediate memory consumption
				val = 0;				
		}
		
		return val;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
		if( _direction == Direction.Col && mc.colsKnown() ) 
			ret = new long[]{1, mc.getCols(), -1};
		else if( _direction == Direction.Row && mc.rowsKnown() )
			ret = new long[]{mc.getRows(), 1, -1};
		
		return ret;
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
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
			{
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
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE ) {
			setRequiresRecompile();
		}
		
		return _etype;
	}
	
	/**
	 * 
	 * @param input
	 * @param dir
	 * @return
	 */
	private boolean requiresAggregation( Hop input, Direction dir ) 
	{
		if( !ALLOW_UNARYAGG_WO_FINAL_AGG )
			return false; //customization not allowed
		
		boolean noAggRequired = 
				  ( input.getDim1()>1 && input.getDim1()<=input.getRowsInBlock() && dir==Direction.Col ) //e.g., colSums(X) with nrow(X)<=1000
				||( input.getDim2()>1 && input.getDim2()<=input.getColsInBlock() && dir==Direction.Row ); //e.g., rowSums(X) with ncol(X)<=1000
	
		return !noAggRequired;
	}
	

	/**
	 * 
	 * @param agg
	 * @return
	 */
	private SparkAggType getSparkUnaryAggregationType( boolean agg )
	{
		if( !agg )
			return SparkAggType.NONE;
		
		if(   getDataType()==DataType.SCALAR //in case of scalars the block dims are not set
		   || dimsKnown() && getDim1()<=getRowsInBlock() && getDim2()<=getColsInBlock() )
			return SparkAggType.SINGLE_BLOCK;
		else
			return SparkAggType.MULTI_BLOCK;
	}
	
	/**
	 * 
	 * @return
	 */
	private boolean isTernaryAggregateRewriteApplicable() 
	{
		boolean ret = false;
		
		//currently we support only sum over binary multiply but potentially 
		//it can be generalized to any RC aggregate over two common binary operations
		if( _direction == Direction.RowCol 
			&& _op == AggOp.SUM ) 
		{
			Hop input1 = getInput().get(0);
			if( input1 instanceof BinaryOp && ((BinaryOp)input1).getOp()==OpOp2.MULT 
				&& input1.getDataType()==DataType.MATRIX&& input1.getDim2()==1 ) //all column vectors
			{
				Hop input11 = input1.getInput().get(0);
				Hop input12 = input1.getInput().get(1);
				
				if( input11 instanceof BinaryOp && ((BinaryOp)input11).getOp()==OpOp2.MULT )
				{
					ret = (input11.getInput().get(0).getDim1()==input1.getDim1() 
						&& input11.getInput().get(1).getDim1()==input1.getDim1()
						&& input12.getDim1()==input1.getDim1());
				}
				else if( input12 instanceof BinaryOp && ((BinaryOp)input12).getOp()==OpOp2.MULT )
				{
					ret = (input12.getInput().get(0).getDim1()==input1.getDim1() 
							&& input12.getInput().get(1).getDim1()==input1.getDim1()
							&& input11.getDim1()==input1.getDim1());
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	private boolean isUnaryAggregateOuterRewriteApplicable() 
	{
		boolean ret = false;
		Hop input = getInput().get(0);
		
		if( input instanceof BinaryOp && ((BinaryOp)input).isOuterVectorOperator() )
		{
			//for special cases, we need to hold the broadcast twice in order to allow for
			//an efficient binary search over a plain java array
			double factor = (((BinaryOp)input).getOp()==OpOp2.LESS 
					&& _direction == Direction.Row && _op == AggOp.SUM) ? 2.0 : 1.0;
			
			//note: memory constraint only needs to take the rhs into account because the output
			//is guaranteed to be an aggregate of <=16KB
			Hop right = input.getInput().get(1);
			if(  (right.dimsKnown() && factor*OptimizerUtils.estimateSize(right.getDim1(), right.getDim2())
				  < OptimizerUtils.getRemoteMemBudgetMap(true)) //dims known and estimate fits
			   ||(!right.dimsKnown() && factor*right.getOutputMemEstimate()
				  <	OptimizerUtils.getRemoteMemBudgetMap(true)))//dims unknown but worst-case estimate fits 
			{
				ret = true;
			}
		}
		
		return ret;
	}
	
	/**
	 * This will check if there is sufficient memory locally (twice the size of second matrix, for original and sort data), and remotely (size of second matrix (sorted data)).  
	 * @return
	 */
	private boolean isUnaryAggregateOuterSPRewriteApplicable() 
	{
		boolean ret = false;
		Hop input = getInput().get(0);
		
		if( input instanceof BinaryOp && ((BinaryOp)input).isOuterVectorOperator() )
		{
			//for special cases, we need to hold the broadcast twice in order to allow for
			//an efficient binary search over a plain java array
			double factor = (( ((BinaryOp)input).getOp()==OpOp2.LESS || ((BinaryOp)input).getOp()==OpOp2.GREATER
							|| ((BinaryOp)input).getOp()==OpOp2.LESSEQUAL || ((BinaryOp)input).getOp()==OpOp2.GREATEREQUAL
							|| ((BinaryOp)input).getOp()==OpOp2.EQUAL || ((BinaryOp)input).getOp()==OpOp2.NOTEQUAL )
					&& _direction == Direction.Row && _op == AggOp.SUM) ? 2.0 : 1.0;
			
			
			//note: memory constraint only needs to take the rhs into account because the output
			//is guaranteed to be an aggregate of <=16KB
			Hop right = input.getInput().get(1);
			if((  (right.dimsKnown() && OptimizerUtils.estimateSize(right.getDim1(), right.getDim2())
				  < SparkExecutionContext.getBroadcastMemoryBudget()) //dims known and estimate fits
			   ||(!right.dimsKnown() && right.getOutputMemEstimate()
				  <	SparkExecutionContext.getBroadcastMemoryBudget()))//dims unknown but worst-case estimate fits
				&&
				  ((right.dimsKnown() && factor*OptimizerUtils.estimateSize(right.getDim1(), right.getDim2())
						  < OptimizerUtils.getLocalMemBudget()) //dims known and estimate fits
					   ||(!right.dimsKnown() && factor*right.getOutputMemEstimate()
						  <	OptimizerUtils.getLocalMemBudget())))//dims unknown but worst-case estimate fits
						
			{
				ret = true;
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructLopsTernaryAggregateRewrite(ExecType et) 
		throws HopsException, LopsException
	{
		Hop input1 = getInput().get(0);
		Hop input11 = input1.getInput().get(0);
		Hop input12 = input1.getInput().get(1);
		
		Lop ret = null;
		Lop in1 = null;
		Lop in2 = null;
		Lop in3 = null;
		
		if( input11 instanceof BinaryOp && ((BinaryOp)input11).getOp()==OpOp2.MULT )
		{
			in1 = input11.getInput().get(0).constructLops();
			in2 = input11.getInput().get(1).constructLops();
			in3 = input12.constructLops();
		}
		else if( input12 instanceof BinaryOp && ((BinaryOp)input12).getOp()==OpOp2.MULT )
		{
			in1 = input11.constructLops();
			in2 = input12.getInput().get(0).constructLops();
			in3 = input12.getInput().get(1).constructLops();
		}
		else 
		{
			throw new HopsException("Failed to apply ternary-aggregate hop-lop rewrite - missing binaryop.");
		}

		//create new ternary aggregate operator 
		ret = new TernaryAggregate(in1, in2, in3, Aggregate.OperationTypes.KahanSum, Binary.OperationTypes.MULTIPLY, DataType.SCALAR, ValueType.DOUBLE, et);
		
		return ret;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if (getDataType() != DataType.SCALAR)
		{
			Hop input = getInput().get(0);
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
                      (_op == AggOp.SUM || _op == AggOp.MIN || //valid aggregration functions
		               _op == AggOp.MAX || _op == AggOp.PROD || 
		               _op == AggOp.MEAN);
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
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
