package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;


public class LiteralOp extends Hops {

	private double value_double;
	private long value_long;
	private String value_string;
	private boolean value_boolean;

	// INT, DOUBLE, STRING, BOOLEAN}

	public LiteralOp(String l, double value) {
		super(Kind.LiteralOp, l, DataType.SCALAR, ValueType.DOUBLE);
		this.value_double = value;
	}

	public LiteralOp(String l, long value) {
		super(Kind.LiteralOp, l, DataType.SCALAR, ValueType.INT);
		this.value_long = value;
	}

	public LiteralOp(String l, String value) {
		super(Kind.LiteralOp, l, DataType.SCALAR, ValueType.STRING);
		this.value_string = value;
	}

	public LiteralOp(String l, boolean value) {
		super(Kind.LiteralOp, l, DataType.SCALAR, ValueType.BOOLEAN);
		this.value_boolean = value;
	}

	@Override
	public Lops constructLops()
			throws HopsException {

		if (get_lops() == null) {

			Lops l = null;

			switch (get_valueType()) {
			case DOUBLE:
				l = new Data(null,
						Data.OperationTypes.READ, null, Double
								.toString(value_double), get_dataType(),
						get_valueType(), false);
				break;
			case BOOLEAN:
				l = new Data(null,
						Data.OperationTypes.READ, null, Boolean
								.toString(value_boolean), get_dataType(),
						get_valueType(), false);
				break;
			case STRING:
				l = new Data(null,
						Data.OperationTypes.READ, null, value_string,
						get_dataType(), get_valueType(), false);
				break;
			case INT:
				l = new Data(null,
						Data.OperationTypes.READ, null, Long.toString(value_long), get_dataType(),
						get_valueType(), false);
				break;
			default:
				throw new HopsException(this.printErrorLocation() + 
						"unexpected value type constructing lops for LiteralOp.\n");

			}

			l.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			l.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			set_lops(l);
		}

		return get_lops();
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			switch (get_valueType()) {
			case DOUBLE:
				System.out.println("  Value: " + value_double + "\n");
				break;
			case BOOLEAN:
				System.out.println("  Value: " + value_boolean + "\n");
				break;
			case STRING:
				System.out.println("  Value: " + value_string + "\n");
				break;
			case INT:
				System.out.println("  Value: " + value_long + "\n");
				break;
			default:
				throw new HopsException(this.printErrorLocation() +
						"unexpected value type printing LiteralOp.\n");
			}

			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
	}

	@Override
	public String getOpString() {
		String val = "";
		switch (get_valueType()) {
		case DOUBLE:
			val = Double.toString(value_double);
			break;
		case BOOLEAN:
			val = Boolean.toString(value_boolean);
			break;
		case STRING:
			val = value_string;
			break;
		case INT:
			val = Long.toString(value_long);
			break;
		}
		return "LiteralOp" + val;
	}
	
	private SQLLopProperties getProperties()
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		
		String val = null;
		switch (get_valueType()) {
		case DOUBLE:
			val = Double.toString(value_double);
			break;
		case BOOLEAN:
			val = Boolean.toString(value_boolean);
			break;
		case STRING:
			val = value_string;
			break;
		case INT:
			val = Long.toString(value_long);
			break;
		}
		
		prop.setOpString(val);
		
		return prop;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		/*
		 * Does not generate SQL, instead the actual value is passed in the table name and can be inserted directly
		 */
		if(this.get_sqllops() == null)
		{
			SQLLops sqllop = new SQLLops(this.get_name(),
										GENERATES.NONE,
										this.get_valueType(),
										this.get_dataType());

			//Retrieve string for value
			if(this.get_valueType() == ValueType.DOUBLE)
				sqllop.set_tableName(String.format(Double.toString(this.value_double)));
			else if(this.get_valueType() == ValueType.INT)
				sqllop.set_tableName(String.format(Long.toString(this.value_long)));
			else if(this.get_valueType() == ValueType.STRING)
				sqllop.set_tableName("'" + this.value_string + "'");
			else if(this.get_valueType() == ValueType.BOOLEAN)
				sqllop.set_tableName(Boolean.toString(this.value_boolean));
			
			sqllop.set_properties(getProperties());
			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}

	@Override
	public double computeMemEstimate() {
		
		switch(this.get_valueType()) {
		case INT:
			_outputMemEstimate = OptimizerUtils.INT_SIZE; break;
		case DOUBLE:
			_outputMemEstimate = OptimizerUtils.DOUBLE_SIZE; break;
		case BOOLEAN:
			_outputMemEstimate = OptimizerUtils.BOOLEAN_SIZE; break;
		case STRING: 
			_outputMemEstimate = this.value_string.length() * OptimizerUtils.CHAR_SIZE; break;
		case OBJECT:
			_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE; break;
		}
		
		_memEstimate = getInputOutputSize();
		return _memEstimate;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		// Since a Literal hop does not represent any computation, 
		// this function is not applicable. 
		return null;
	}
}
