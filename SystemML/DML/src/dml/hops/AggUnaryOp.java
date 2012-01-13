package dml.hops;

import dml.lops.Aggregate;
import dml.lops.Group;
import dml.lops.Lops;
import dml.lops.PartialAggregate;
import dml.lops.UnaryCP;
import dml.lops.LopProperties.ExecType;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.sql.sqllops.SQLCondition;
import dml.sql.sqllops.SQLLopProperties;
import dml.sql.sqllops.SQLLops;
import dml.sql.sqllops.SQLSelectStatement;
import dml.sql.sqllops.SQLTableReference;
import dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import dml.sql.sqllops.SQLLops.GENERATES;
import dml.utils.HopsException;

/* Aggregate unary (cell) operation: Sum (aij), col_sum, row_sum
 * 		Properties: 
 * 			Symbol: +, min, max, ...
 * 			1 Operand
 * 	
 * 		Semantic: generate indices, align, aggregate
 */

public class AggUnaryOp extends Hops {

	private AggOp _op;
	private Direction _direction;

	public AggUnaryOp(String l, DataType dt, ValueType vt, AggOp o,
			Direction idx, Hops inp) {
		super(Kind.AggUnaryOp, l, dt, vt);
		_op = o;
		_direction = idx;
		getInput().add(0, inp);
	
		inp.getParent().add(this);
	}

	public Lops constructLops()
			throws HopsException {
		if (get_lops() == null) {
			try {
				PartialAggregate transform1 = new PartialAggregate(
						getInput().get(0).constructLops(), HopsAgg2Lops
								.get(_op), HopsDirection2Lops.get(_direction),
						get_dataType(), get_valueType());
				transform1.setDimensionsBasedOnDirection(get_dim1(),
						get_dim2(), get_rows_per_block(), get_cols_per_block());

				Group group1 = new Group(
						transform1, Group.OperationTypes.Sort, get_dataType(),
						get_valueType());
				group1.getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_per_block(), get_cols_per_block());

				Aggregate agg1 = new Aggregate(
						group1, HopsAgg2Lops.get(_op), get_dataType(),
						get_valueType(), ExecType.MR);
				agg1.getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_per_block(), get_cols_per_block());
				agg1.setupCorrectionLocation(transform1.getCorrectionLocaion());

				set_lops(agg1);

				if (get_dataType() == DataType.SCALAR) {
					// In case of SUM(), an explicit cast must be performed so
					// that the result is stored in a scalar variable
					transform1.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(), get_rows_per_block(),
							get_cols_per_block());

					// Set the dimensions of PartialAggregate LOP based on the
					// direction in which aggregation is performed
					transform1.setDimensionsBasedOnDirection(getInput().get(0)
							.get_dim1(), getInput().get(0).get_dim2(),
							get_rows_per_block(), get_cols_per_block());
					group1.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(), get_rows_per_block(),
							get_cols_per_block());
					agg1.getOutputParameters().setDimensions(1, 1,
							get_rows_per_block(), get_cols_per_block());
					UnaryCP unary1 = new UnaryCP(
							agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
							get_dataType(), get_valueType());
					set_lops(unary1);

				}
			} catch (Exception e) {
				throw new HopsException(e);
			}

		}
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "a(" + HopsAgg2String.get(_op)
				+ HopsDirection2String.get(_direction) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println("  Operation: " + _op);
			System.out.println("  Direction: " + _direction + "\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		
		if(this.get_sqllops() == null)
		{
			if(this.getInput().size() != 1)
				throw new HopsException("The aggregate unary hop must have one input");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			Hops input = this.getInput().get(0);
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
	
	private SQLSelectStatement getSQLSelect(Hops input)
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
	
	private SQLLopProperties getProperties(Hops input)
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
		prop.setOpString(Hops.HopsAgg2String.get(_op) + " " + input.get_sqllops().get_tableName());
		
		return prop;
	}
	
	private String getSQLSelectCode(Hops input)
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
			sql = String.format(SQLLops.UNARYMAXMIN, Hops.HopsAgg2String.get(this._op),input.get_sqllops().get_tableName());
		}
		return sql;
	}
}
