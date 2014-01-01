/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
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


/* Unary (cell operations): aij = -b
 * 		Semantic: given a value, perform the operation (in mapper or reducer)
 */

public class UnaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum Position {
		before, after, none
	};

	private OpOp1 _op = null;

	
	private UnaryOp() {
		//default constructor for clone
	}
	
	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hop inp)
			throws HopsException {
		super(Hop.Kind.UnaryOp, l, dt, vt);

		getInput().add(0, inp);
		inp.getParent().add(this);

		_op = o;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	// this is for OpOp1, e.g. A = -B (0-B); and a=!b
	public OpOp1 get_op() {
		return _op;
	}
	
	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "u(" + _op + ")";
		// s += HopsOpOp1String.get(_op) + ")";
		return s;
	}

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{		
		if (get_lops() == null) {
			try {
			if (get_dataType() == DataType.SCALAR) {
				if (_op == Hop.OpOp1.IQM) {
					ExecType et = optFindExecType();
					if ( et == ExecType.MR ) {
						CombineUnary combine = CombineUnary
								.constructCombineLop(
										getInput()
												.get(0).constructLops(),
										DataType.MATRIX, get_valueType());
						combine.getOutputParameters().setDimensions(
								getInput().get(0).get_dim1(),
								getInput().get(0).get_dim2(), 
								getInput().get(0).get_rows_in_block(),
								getInput().get(0).get_cols_in_block(),
								getInput().get(0).getNnz());
	
						SortKeys sort = SortKeys
								.constructSortByValueLop(
										combine,
										SortKeys.OperationTypes.WithNoWeights,
										DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
	
						// Sort dimensions are same as the first input
						sort.getOutputParameters().setDimensions(
								getInput().get(0).get_dim1(),
								getInput().get(0).get_dim2(),
								getInput().get(0).get_rows_in_block(),
								getInput().get(0).get_cols_in_block(),
								getInput().get(0).getNnz());
	
						Data lit = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.25));
						
						lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	                    			
						PickByCount pick = new PickByCount(
								sort, lit, DataType.MATRIX, get_valueType(),
								PickByCount.OperationTypes.RANGEPICK);
	
						pick.getOutputParameters().setDimensions(-1, -1,  
								get_rows_in_block(), get_cols_in_block(), -1);
						
						pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						PartialAggregate pagg = new PartialAggregate(
								pick, HopsAgg2Lops.get(Hop.AggOp.SUM),
								HopsDirection2Lops.get(Hop.Direction.RowCol),
								DataType.MATRIX, get_valueType());
						
						pagg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						// Set the dimensions of PartialAggregate LOP based on the
						// direction in which aggregation is performed
						pagg.setDimensionsBasedOnDirection(get_dim1(),
									get_dim2(), get_rows_in_block(),
									get_cols_in_block());
	
						Group group1 = new Group(
								pagg, Group.OperationTypes.Sort, DataType.MATRIX,
								get_valueType());
						group1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						Aggregate agg1 = new Aggregate(
								group1, HopsAgg2Lops.get(Hop.AggOp.SUM),
								DataType.MATRIX, get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						agg1.setupCorrectionLocation(pagg.getCorrectionLocaion());
						
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						UnaryCP unary1 = new UnaryCP(
								agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
								get_dataType(), get_valueType());
						unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
						unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						BinaryCP binScalar1 = new BinaryCP(
								sort, lit, BinaryCP.OperationTypes.IQSIZE,
								DataType.SCALAR, get_valueType());
						binScalar1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
	
						binScalar1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						BinaryCP binScalar2 = new BinaryCP(
								unary1, binScalar1, HopsOpOp2LopsBS
										.get(Hop.OpOp2.DIV), DataType.SCALAR,
								get_valueType());
						binScalar2.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
						
						binScalar2.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						set_lops(binScalar2);
					}
					else {
						SortKeys sort = SortKeys.constructSortByValueLop(
								getInput().get(0).constructLops(), 
								SortKeys.OperationTypes.WithNoWeights, 
								DataType.MATRIX, ValueType.DOUBLE, et );
						sort.getOutputParameters().setDimensions(
								getInput().get(0).get_dim1(),
								getInput().get(0).get_dim2(),
								getInput().get(0).get_rows_in_block(),
								getInput().get(0).get_cols_in_block(),
								getInput().get(0).getNnz());
						PickByCount pick = new PickByCount(
								sort,
								null,
								get_dataType(),
								get_valueType(),
								PickByCount.OperationTypes.IQM, et, true);
			
						pick.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			
						pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(pick);
					}
				} else {
					UnaryCP unary1 = new UnaryCP(
							getInput().get(0).constructLops(), HopsOpOp1LopsUS
									.get(_op), get_dataType(), get_valueType());
					unary1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
							             get_rows_in_block(), get_cols_in_block(), getNnz());
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(unary1);
				}

			} else {
				ExecType et = optFindExecType();
				Unary unary1 = new Unary(
						getInput().get(0).constructLops(), HopsOpOp1LopsU
								.get(_op), get_dataType(), get_valueType(), et);
				unary1.getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
				unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				set_lops(unary1);
			}
			} catch (Exception e) {
				throw new HopsException(this.printErrorLocation() + "error constructing Lops for UnaryOp Hop -- \n " , e);
			}
		}
		
		return get_lops();
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		
		if(this.get_sqllops() == null)
		{
			if(this.getInput().size() < 1)
				throw new HopsException(this.printErrorLocation() + "Unary hop needs one input \n");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			if(this._op == OpOp1.PRINT2 || this._op == OpOp1.PRINT)
				gen = GENERATES.PRINT;
			else if(this.get_dataType() == DataType.SCALAR && gen != GENERATES.DML_PERSISTENT 
					&& gen != GENERATES.DML_TRANSIENT)
				gen = GENERATES.DML;
			
			Hop input = this.getInput().get(0);
			
			SQLLops sqllop = new SQLLops(this.get_name(),
										gen,
										input.constructSQLLOPs(),
										this.get_valueType(),
										this.get_dataType());

			//TODO Uncomment this to make scalar placeholders
			if(this.get_dataType() == DataType.SCALAR && gen == GENERATES.DML)
				sqllop.set_tableName("##" + sqllop.get_tableName() + "##");
			
			String sql = this.getSQLSelectCode(input);
			sqllop.set_sql(sql);
			sqllop.set_properties(getProperties(input));
			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}
	
	private SQLLopProperties getProperties(Hop input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		
		prop.setOpString(Hop.HopsOpOp12String.get(this._op) + "(" + input.get_sqllops().get_tableName() + ")");
		
		return prop;
	}
	
	
	
	private String getSQLSelectCode(Hop input) throws HopsException
	{
		String sql = null;

		if(input.get_sqllops().get_dataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				//sqllop.set_dataType(DataType.SCALAR);
				sql = String.format(SQLLops.CASTASSCALAROP, input.get_sqllops().get_tableName());
			}
			if(Hop.isFunction(this._op))
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				if(this._op == OpOp1.LOG)
					op = "ln";
				sql = String.format(SQLLops.UNARYFUNCOP, op, input.get_sqllops().get_tableName());
			}
			//Currently there is only minus that is just added in front of the operand
			else if(this._op == OpOp1.MINUS)
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				sql = String.format(SQLLops.UNARYOP, op, input.get_sqllops().get_tableName());
			}
			//Not is in SLQ not "!" but "NOT"
			else if(this._op == OpOp1.NOT)
			{
				sql = String.format(SQLLops.UNARYNOT, input.get_sqllops().get_tableName());
			}
		}
		
		else if(input.get_sqllops().get_dataType() == DataType.SCALAR)
		{
			String table = "";
			String opr = null;
			String s = input.get_sqllops().get_tableName();
			
			if(input.get_sqllops().get_flag() == GENERATES.SQL)
			{
				String squoted = SQLLops.addQuotes(s);
				table = " FROM " + squoted;
				s = squoted + "." + SQLLops.SCALARVALUECOLUMN;
			}
			if(this._op == OpOp1.PRINT2 || this._op == OpOp1.PRINT)
			{
				//sqllop.set_flag(GENERATES.PRINT);
				String tname = input.get_sqllops().get_tableName();
				
				if(input.get_sqllops().get_dataType() == DataType.MATRIX
						|| input.get_sqllops().get_flag() == GENERATES.SQL)
				{
					tname = SQLLops.addQuotes(tname);
				
					sql = String.format(SQLLops.SIMPLESCALARSELECTFROM,
						tname + "." + SQLLops.SCALARVALUECOLUMN, tname);
				}
				else
					sql = tname;
			}
			else
			{
				//Create correct string for operation
				if(Hop.isFunction(this._op))
					opr = Hop.HopsOpOp12String.get(this._op) + "( " + s + " )";
				else if(this._op == OpOp1.MINUS)
					opr = Hop.HopsOpOp12String.get(this._op) + s;
				else if(this._op == OpOp1.NOT)
					opr = "NOT " + s;

				sql = String.format(SQLLops.UNARYSCALAROP + table, opr);
			}
		}
		else
			throw new HopsException(this.printErrorLocation() + "Other unary operations than matrix and scalar operations are currently not supported \n");
		return sql;
	}
	
	private SQLSelectStatement getSQLSelect(Hop input) throws HopsException
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		
		if(input.get_sqllops().get_dataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				stmt.getColumns().add("max(value) AS sval");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
			if(Hop.isFunction(this._op))
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				//SELECT %s(alias_a.value) as value FROM \"%s\" alias_a
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add(op + "(value) AS value");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
			//Currently there is only minus that is just added in front of the operand
			else if(this._op == OpOp1.MINUS)
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				//SELECT alias_a.row AS row, alias_a.col AS col, %salias_a.value FROM \"%s\" alias_a";
				
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add(op + "value AS value");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
			//Not is in SLQ not "!" but "NOT"
			else if(this._op == OpOp1.NOT)
			{
				//SELECT alias_a.row AS row, alias_a.col AS col, 1 as value FROM \"%s\" alias_a WHERE alias_a.value == 0";
				
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add("1 AS value");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
				stmt.getWheres().add(new SQLCondition("value == 0"));
			}
		}
		
		else if(input.get_sqllops().get_dataType() == DataType.SCALAR)
		{
			String opr = null;

			if(this._op == OpOp1.PRINT2 || this._op == OpOp1.PRINT)
			{
				//sqllop.set_flag(GENERATES.PRINT);
				String tname = input.get_sqllops().get_tableName();
				
				if(input.get_sqllops().get_dataType() == DataType.MATRIX
						|| input.get_sqllops().get_flag() == GENERATES.SQL)
				{
					tname = SQLLops.addQuotes(tname);
				
					//SELECT %s AS sval FROM %s
					
					stmt.getColumns().add(SQLLops.SCALARVALUECOLUMN);
					stmt.setTable(new SQLTableReference(tname));
				}
				//else
				//	sql = tname;
			}
			else
			{
				//Create correct string for operation
				if(Hop.isFunction(this._op))
					opr = Hop.HopsOpOp12String.get(this._op) + "( sval )";
				else if(this._op == OpOp1.MINUS)
					opr = Hop.HopsOpOp12String.get(this._op) + "sval";
				else if(this._op == OpOp1.NOT)
					opr = "NOT sval";
				
				stmt.getColumns().add(opr + " AS sval");
				if(input.get_sqllops().get_flag() == GENERATES.SQL)
					stmt.setTable(new SQLTableReference(SQLLops.addQuotes(input.get_sqllops().get_tableName())));
			}
		}
		else
			throw new HopsException(this.printErrorLocation() + "Other unary operations than matrix and scalar operations are currently not supported \n");
		return stmt;
	}
	
	@Override
	public void computeMemEstimate(MemoTable memo)
	{
		//overwrites default hops behavior
		super.computeMemEstimate(memo);
		
		if( _op == Hop.OpOp1.NROW || _op == Hop.OpOp1.NCOL ) //specific case for meta data ops
		{
			_memEstimate = OptimizerUtils.INT_SIZE;
			//_outputMemEstimate = OptimizerUtils.INT_SIZE;
			//_processingMemEstimate = 0;
		}
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
		double ret = 0;
		
		if ( _op == OpOp1.IQM ) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation
			// getMemEstimate works for both cases of known dims and worst-case stats
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
		if( mc.dimsKnown() ) {
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN 
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN  
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
			{
				ret = new long[]{mc.get_rows(), mc.get_cols(), mc.getNonZeros()};
			}
			else 
				ret = new long[]{mc.get_rows(), mc.get_cols(), -1};	
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
	
		if( _etypeForced != null ) 			
			_etype = _etypeForced;		
		else 
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
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
		if ( get_dataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else 
		{
			// If output is a Matrix then this operation is of type (B = op(A))
			// Dimensions of B are same as that of A, and sparsity may/maynot change
			Hop input = getInput().get(0);
			set_dim1( input.get_dim1() );
			set_dim2( input.get_dim2() );
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN  
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
			{
				setNnz( input.getNnz() );
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		UnaryOp ret = new UnaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.UnaryOp )
			return false;
		
		UnaryOp that2 = (UnaryOp)that;		
		return (   _op == that2._op
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
