/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.TernaryAggregate;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;


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
		super(Kind.AggUnaryOp, l, dt, vt);
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
					agg1 = constructLopsTernaryAggregateRewrite();
				}
				else { //general case
					agg1 = new PartialAggregate(input.constructLops(), 
							HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), getDataType(),getValueType(), et);
				}
				
				agg1.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(agg1);
				setLops(agg1);
				
				if (getDataType() == DataType.SCALAR) {
					agg1.getOutputParameters().setDimensions(1, 1, getRowsInBlock(), getColsInBlock(), getNnz());
				}
			}
			else if( et == ExecType.MR )
			{
				//unary aggregate
				PartialAggregate transform1 = new PartialAggregate(input.constructLops(), 
						HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), DataType.MATRIX, getValueType());
				transform1.setDimensionsBasedOnDirection(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock());
				setLineNumbers(transform1);
				
				Lop aggregate = null;
				Group group1 = null; 
				Aggregate agg1 = null;
				if( requiresAggregation(input, _direction) )
				{
					group1 = new Group(transform1, Group.OperationTypes.Sort, DataType.MATRIX, getValueType());
					group1.getOutputParameters().setDimensions(getDim1(), getDim2(),getRowsInBlock(), getColsInBlock(), getNnz());
					setLineNumbers(group1);
					
					agg1 = new Aggregate(group1, HopsAgg2Lops.get(_op), DataType.MATRIX, getValueType(), et);
					agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
					agg1.setupCorrectionLocation(transform1.getCorrectionLocation());
					setLineNumbers(agg1);
					
					aggregate = agg1;
				}
				else
				{
					transform1.setDropCorrection();
					aggregate = transform1;
				}
				
				setLops(aggregate);

				if (getDataType() == DataType.SCALAR) {

					// Set the dimensions of PartialAggregate LOP based on the
					// direction in which aggregation is performed
					transform1.setDimensionsBasedOnDirection(input.getDim1(), input.getDim2(),
							getRowsInBlock(), getColsInBlock());
					
					if( group1 != null && agg1 != null ) { //if aggregation required
						group1.getOutputParameters().setDimensions(input.getDim1(), input.getDim2(), 
								getRowsInBlock(), getColsInBlock(), getNnz());
						agg1.getOutputParameters().setDimensions(1, 1, 
								getRowsInBlock(), getColsInBlock(), getNnz());
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
				boolean needAgg = requiresAggregation(input, _direction);
				
				//unary aggregate
				PartialAggregate transform1 = new PartialAggregate(input.constructLops(), 
						HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), DataType.MATRIX, getValueType(), needAgg, et);
				transform1.setDimensionsBasedOnDirection(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock());
				setLineNumbers(transform1);
				setLops(transform1);

				if (getDataType() == DataType.SCALAR) {
					UnaryCP unary1 = new UnaryCP(transform1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
							                    getDataType(), getValueType());
					unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					setLineNumbers(unary1);
					setLops(unary1);
				}
			}
		} 
		catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "In AggUnary Hop, error constructing Lops " , e);
		}
		
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
	public SQLLops constructSQLLOPs() throws HopsException {
		
		if(this.getSqlLops() == null)
		{
			if(this.getInput().size() != 1)
				throw new HopsException(this.printErrorLocation() + "The aggregate unary hop must have one input");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			Hop input = this.getInput().get(0);
			SQLLops lop = new SQLLops(this.getName(),
									gen,
									input.constructSQLLOPs(),
									this.getValueType(),
									this.getDataType());
			
			//TODO Uncomment this to make scalar placeholders
			if(this.getDataType() == DataType.SCALAR && gen == GENERATES.DML)
				lop.set_tableName("##" + lop.get_tableName() + "##");
			
			lop.set_sql(getSQLSelectCode(input));
			lop.set_properties(getProperties(input));
			this.setSqlLops(lop);
			return lop;
		}
		else
			return this.getSqlLops();
	}
	
	@SuppressWarnings("unused")
	private SQLSelectStatement getSQLSelect(Hop input)
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		stmt.setTable(new SQLTableReference(SQLLops.addQuotes(input.getSqlLops().get_tableName()), ""));
		
		if((this._op == AggOp.SUM && _direction == Direction.RowCol) || this._op == AggOp.TRACE)
		{
			stmt.getColumns().add("SUM(value)");
			if(this._op == AggOp.TRACE)
				stmt.getWheres().add(new SQLCondition("row = col"));
		}
		else if(_op == AggOp.SUM)
		{
			if(_direction == Direction.Row)
			{
				stmt.getColumns().add("row");
				stmt.getColumns().add("1 AS col");
				stmt.getGroupBys().add("row");
			}
			else
			{
				stmt.getColumns().add("1 AS row");
				stmt.getColumns().add("col");
				stmt.getGroupBys().add("col");
			}
			stmt.getColumns().add("SUM(value)");
		}
		
		return stmt;
	}
	
	private SQLLopProperties getProperties(Hop input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		JOINTYPE join = JOINTYPE.NONE;
		AGGREGATIONTYPE agg = AGGREGATIONTYPE.NONE;
		
		//TODO: PROD
		if(_op == AggOp.SUM || _op == AggOp.TRACE)
			agg = AGGREGATIONTYPE.SUM;
		else if(_op == AggOp.MAX)
			agg = AGGREGATIONTYPE.MAX;
		else if(_op == AggOp.MIN)
			agg = AGGREGATIONTYPE.MIN;
		
		prop.setAggType(agg);
		prop.setJoinType(join);
		prop.setOpString(Hop.HopsAgg2String.get(_op) + " " + input.getSqlLops().get_tableName());
		
		return prop;
	}
	
	private String getSQLSelectCode(Hop input)
	{
		String sql = null;
		if(this._op == AggOp.PROD)
		{
		
		}
		else if(this._op == AggOp.TRACE)
		{
			sql = String.format(SQLLops.UNARYTRACE, input.getSqlLops().get_tableName());
		}
		else if(this._op == AggOp.SUM)
		{
			if(this._direction == Direction.RowCol)
				sql = String.format(SQLLops.UNARYSUM, input.getSqlLops().get_tableName());
			else if(this._direction == Direction.Row)
				sql = String.format(SQLLops.UNARYROWSUM, input.getSqlLops().get_tableName());
			else
				sql = String.format(SQLLops.UNARYCOLSUM, input.getSqlLops().get_tableName());
		}
		else
		{
			sql = String.format(SQLLops.UNARYMAXMIN, Hop.HopsAgg2String.get(this._op),input.getSqlLops().get_tableName());
		}
		return sql;
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
				_etype = ExecType.MR;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
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
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructLopsTernaryAggregateRewrite() 
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
		ret = new TernaryAggregate(in1, in2, in3, Aggregate.OperationTypes.KahanSum, Binary.OperationTypes.MULTIPLY, DataType.SCALAR, ValueType.DOUBLE);
		
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
		if( that._kind!=Kind.AggUnaryOp )
			return false;
		
		AggUnaryOp that2 = (AggUnaryOp)that;		
		return (   _op == that2._op
				&& _direction == that2._direction
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
