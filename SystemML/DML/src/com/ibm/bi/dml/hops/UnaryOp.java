package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;


/* Unary (cell operations): aij = -b
 * 		Semantic: given a value, perform the operation (in mapper or reducer)
 */

public class UnaryOp extends Hops {

	public enum Position {
		before, after, none
	};

	private OpOp1 _op = null;

	
	
	// this is for OpOp1, e.g. A = -B (0-B); and a=!b

	public OpOp1 get_op() {
		return _op;
	}

	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hops inp)
			throws HopsException {
		super(Hops.Kind.UnaryOp, l, dt, vt);

		getInput().add(0, inp);
		inp.getParent().add(this);

		_op = o;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				for (Hops h : getInput()) {
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

	public Lops constructLops()
			throws HopsException {
		if (get_lops() == null) {
			try {
			if (get_dataType() == DataType.SCALAR) {
				if (_op == Hops.OpOp1.IQM) {
					ExecType et = optFindExecType();
					if ( et == ExecType.MR ) {
						CombineUnary combine = CombineUnary
								.constructCombineLop(
										(Lops) getInput()
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
										(Lops) combine,
										SortKeys.OperationTypes.WithNoWeights,
										DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
	
						// Sort dimensions are same as the first input
						sort.getOutputParameters().setDimensions(
								getInput().get(0).get_dim1(),
								getInput().get(0).get_dim2(),
								getInput().get(0).get_rows_in_block(),
								getInput().get(0).get_cols_in_block(),
								getInput().get(0).getNnz());
	
						Data lit = new Data(
								null, Data.OperationTypes.READ, null, Double
										.toString(0.25), DataType.SCALAR,
								get_valueType(), false);
						
						lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	                    			
						PickByCount pick = new PickByCount(
								sort, lit, DataType.MATRIX, get_valueType(),
								PickByCount.OperationTypes.RANGEPICK);
	
						pick.getOutputParameters().setDimensions(-1, -1,  
								get_rows_in_block(), get_cols_in_block(), -1);
						
						pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						PartialAggregate pagg = new PartialAggregate(
								pick, HopsAgg2Lops.get(Hops.AggOp.SUM),
								HopsDirection2Lops.get(Hops.Direction.RowCol),
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
								group1, HopsAgg2Lops.get(Hops.AggOp.SUM),
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
										.get(Hops.OpOp2.DIV), DataType.SCALAR,
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
			
			Hops input = this.getInput().get(0);
			
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
	
	private SQLLopProperties getProperties(Hops input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		
		prop.setOpString(Hops.HopsOpOp12String.get(this._op) + "(" + input.get_sqllops().get_tableName() + ")");
		
		return prop;
	}
	
	
	
	private String getSQLSelectCode(Hops input) throws HopsException
	{
		String sql = null;

		if(input.get_sqllops().get_dataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				//sqllop.set_dataType(DataType.SCALAR);
				sql = String.format(SQLLops.CASTASSCALAROP, input.get_sqllops().get_tableName());
			}
			if(Hops.isFunction(this._op))
			{
				String op = Hops.HopsOpOp12String.get(this._op);
				if(this._op == OpOp1.LOG)
					op = "ln";
				sql = String.format(SQLLops.UNARYFUNCOP, op, input.get_sqllops().get_tableName());
			}
			//Currently there is only minus that is just added in front of the operand
			else if(this._op == OpOp1.MINUS)
			{
				String op = Hops.HopsOpOp12String.get(this._op);
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
				if(Hops.isFunction(this._op))
					opr = Hops.HopsOpOp12String.get(this._op) + "( " + s + " )";
				else if(this._op == OpOp1.MINUS)
					opr = Hops.HopsOpOp12String.get(this._op) + s;
				else if(this._op == OpOp1.NOT)
					opr = "NOT " + s;

				sql = String.format(SQLLops.UNARYSCALAROP + table, opr);
			}
		}
		else
			throw new HopsException(this.printErrorLocation() + "Other unary operations than matrix and scalar operations are currently not supported \n");
		return sql;
	}
	
	private SQLSelectStatement getSQLSelect(Hops input) throws HopsException
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		
		if(input.get_sqllops().get_dataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				stmt.getColumns().add("max(value) AS sval");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
			if(Hops.isFunction(this._op))
			{
				String op = Hops.HopsOpOp12String.get(this._op);
				//SELECT %s(alias_a.value) as value FROM \"%s\" alias_a
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add(op + "(value) AS value");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
			//Currently there is only minus that is just added in front of the operand
			else if(this._op == OpOp1.MINUS)
			{
				String op = Hops.HopsOpOp12String.get(this._op);
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
				if(Hops.isFunction(this._op))
					opr = Hops.HopsOpOp12String.get(this._op) + "( sval )";
				else if(this._op == OpOp1.MINUS)
					opr = Hops.HopsOpOp12String.get(this._op) + "sval";
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
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	//MINUS, NOT, ABS, SIN, COS, TAN, SQRT, LOG, EXP, CAST_AS_SCALAR, PRINT, EIGEN, NROW, NCOL, LENGTH, ROUND, IQM, PRINT2
	@Override
	public double computeMemEstimate() {
		
		if ( get_dataType() == DataType.SCALAR ) {
			_outputMemEstimate = OptimizerUtils.DOUBLE_SIZE;
		}
		else {
			// If output is a Matrix then this operation is of type (B = op(A))
			// Dimensions of B are same as that of A, and sparsity may/maynot change
			// The size of input is a good estimate 
			if ( dimsKnown() )
				_outputMemEstimate = OptimizerUtils.estimateSize(_dim1, _dim2, getInput().get(0).getSparsity());
			else 
				_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
		}
		_memEstimate = getInputOutputSize();
		return _memEstimate;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
	
		if( _etypeForced != null ) 			
			_etype = _etypeForced;		
		else 
		{
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() )
				setRequiresRecompile();
			
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
				_etype = ExecType.CP;
			else 
				_etype = ExecType.MR;
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
			Hops input = getInput().get(0);
			set_dim1( input.get_dim1() );
			set_dim2( input.get_dim2() );
		}
	}
}
