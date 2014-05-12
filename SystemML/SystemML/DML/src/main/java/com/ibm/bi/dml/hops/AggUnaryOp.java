/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{	
		if (get_lops() == null) {
			try {
				ExecType et = optFindExecType();
				if ( et == ExecType.CP ) {
					PartialAggregate agg1 = new PartialAggregate(
							getInput().get(0).constructLops(), HopsAgg2Lops.get(_op), HopsDirection2Lops.get(_direction), get_dataType(),
							get_valueType(), et);
					agg1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(agg1);
					if (get_dataType() == DataType.SCALAR) {
						agg1.getOutputParameters().setDimensions(1, 1, 
								get_rows_in_block(), get_cols_in_block(), getNnz());
					}
				}
				else {
					PartialAggregate transform1 = new PartialAggregate(
							getInput().get(0).constructLops(), HopsAgg2Lops
									.get(_op), HopsDirection2Lops.get(_direction),
							DataType.MATRIX, get_valueType());
					
					transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					transform1.setDimensionsBasedOnDirection(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block());
	
					Group group1 = new Group(
							transform1, Group.OperationTypes.Sort, DataType.MATRIX,
							get_valueType());
					group1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(),get_rows_in_block(), get_cols_in_block(), getNnz());
	
					group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					Aggregate agg1 = new Aggregate(
							group1, HopsAgg2Lops.get(_op), DataType.MATRIX,
							get_valueType(), et);
					agg1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					agg1.setupCorrectionLocation(transform1.getCorrectionLocation());
					
					agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(agg1);
	
					if (get_dataType() == DataType.SCALAR) {
	
						// Set the dimensions of PartialAggregate LOP based on the
						// direction in which aggregation is performed
						transform1.setDimensionsBasedOnDirection(getInput().get(0)
								.get_dim1(), getInput().get(0).get_dim2(),
								 get_rows_in_block(), get_cols_in_block());
						group1.getOutputParameters().setDimensions(
								getInput().get(0).get_dim1(),
								getInput().get(0).get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						agg1.getOutputParameters().setDimensions(1, 1, 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						UnaryCP unary1 = new UnaryCP(
								agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
								get_dataType(), get_valueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
						unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						set_lops(unary1);
	
					}
				}
			} catch (Exception e) {
				throw new HopsException(this.printErrorLocation() + "In AggUnary Hop, error constructing Lops " , e);
			}

		}
		
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "au(" + HopsAgg2String.get(_op)
				+ HopsDirection2String.get(_direction) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				LOG.debug("  Direction: " + _direction);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		
		if(this.get_sqllops() == null)
		{
			if(this.getInput().size() != 1)
				throw new HopsException(this.printErrorLocation() + "The aggregate unary hop must have one input");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			Hop input = this.getInput().get(0);
			SQLLops lop = new SQLLops(this.get_name(),
									gen,
									input.constructSQLLOPs(),
									this.get_valueType(),
									this.get_dataType());
			
			//TODO Uncomment this to make scalar placeholders
			if(this.get_dataType() == DataType.SCALAR && gen == GENERATES.DML)
				lop.set_tableName("##" + lop.get_tableName() + "##");
			
			lop.set_sql(getSQLSelectCode(input));
			lop.set_properties(getProperties(input));
			this.set_sqllops(lop);
			return lop;
		}
		else
			return this.get_sqllops();
	}
	
	private SQLSelectStatement getSQLSelect(Hop input)
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		stmt.setTable(new SQLTableReference(SQLLops.addQuotes(input.get_sqllops().get_tableName()), ""));
		
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
		prop.setOpString(Hop.HopsAgg2String.get(_op) + " " + input.get_sqllops().get_tableName());
		
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
			sql = String.format(SQLLops.UNARYTRACE, input.get_sqllops().get_tableName());
		}
		else if(this._op == AggOp.SUM)
		{
			if(this._direction == Direction.RowCol)
				sql = String.format(SQLLops.UNARYSUM, input.get_sqllops().get_tableName());
			else if(this._direction == Direction.Row)
				sql = String.format(SQLLops.UNARYROWSUM, input.get_sqllops().get_tableName());
			else
				sql = String.format(SQLLops.UNARYCOLSUM, input.get_sqllops().get_tableName());
		}
		else
		{
			sql = String.format(SQLLops.UNARYMAXMIN, Hop.HopsAgg2String.get(this._op),input.get_sqllops().get_tableName());
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
			ret = new long[]{1, mc.get_cols(), -1};
		else if( _direction == Direction.Row && mc.rowsKnown() )
			ret = new long[]{mc.get_rows(), 1, -1};
		
		return ret;
	}
	

	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else
		{
			
			
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
			{
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
				_etype = ExecType.CP;
			else 
				_etype = ExecType.MR;
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if (get_dataType() != DataType.SCALAR)
		{
			Hop input = getInput().get(0);
			if ( _direction == Direction.Col ) //colwise computations
			{
				set_dim1(1);
				set_dim2(input.get_dim2());
			}
			else if ( _direction == Direction.Row )
			{
				set_dim1(input.get_dim1());
				set_dim2(1);	
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
