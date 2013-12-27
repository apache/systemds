/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;


public class LiteralOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private double value_double = Double.NaN;
	private long value_long = Long.MAX_VALUE;
	private String value_string;
	private boolean value_boolean;

	// INT, DOUBLE, STRING, BOOLEAN}

	private LiteralOp() {
		//default constructor for clone
	}
	
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
	public Lop constructLops()
			throws HopsException {

		if (get_lops() == null) {

			try {
			Lop l = null;

			switch (get_valueType()) {
			case DOUBLE:
				l = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(value_double));
				break;
			case BOOLEAN:
				l = Data.createLiteralLop(ValueType.BOOLEAN, Boolean.toString(value_boolean));
				break;
			case STRING:
				l = Data.createLiteralLop(ValueType.STRING, value_string);
				break;
			case INT:
				l = Data.createLiteralLop(ValueType.INT, Long.toString(value_long));
				break;
			default:
				throw new HopsException(this.printErrorLocation() + 
						"unexpected value type constructing lops for LiteralOp.\n");

			}

			l.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			l.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			set_lops(l);
			} catch(LopsException e) {
				throw new HopsException(e);
			}
		}

		return get_lops();
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				switch (get_valueType()) {
				case DOUBLE:
					LOG.debug("  Value: " + value_double);
					break;
				case BOOLEAN:
					LOG.debug("  Value: " + value_boolean);
					break;
				case STRING:
					LOG.debug("  Value: " + value_string);
					break;
				case INT:
					LOG.debug("  Value: " + value_long);
					break;
				default:
					throw new HopsException(this.printErrorLocation() +
							"unexpected value type printing LiteralOp.\n");
				}

				for (Hop h : getInput()) {
					h.printMe();
				}

			}
			set_visited(VISIT_STATUS.DONE);
		}
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
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		switch(this.get_valueType()) 
		{
			case INT:
				ret = OptimizerUtils.INT_SIZE; break;
			case DOUBLE:
				ret = OptimizerUtils.DOUBLE_SIZE; break;
			case BOOLEAN:
				ret = OptimizerUtils.BOOLEAN_SIZE; break;
			case STRING: 
				ret = this.value_string.length() * OptimizerUtils.CHAR_SIZE; break;
			case OBJECT:
				ret = OptimizerUtils.DEFAULT_SIZE; break;
		}
		
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		return null;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}	
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		// Since a Literal hop does not represent any computation, 
		// this function is not applicable. 
		return null;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		//do nothing; it is a scalar
	}
	
	public long getLongValue() throws HopsException {
		if ( get_valueType() == ValueType.INT ) {
			return value_long;
		}
		else if ( get_valueType() == ValueType.DOUBLE ) 
			return UtilFunctions.toLong(value_double);
		else
			throw new HopsException("Can not coerce an object of type " + get_valueType() + " into Long.");
	}
	
	public double getDoubleValue() throws HopsException {
		if ( get_valueType() == ValueType.INT ) {
			return value_long;
		}
		else if ( get_valueType() == ValueType.DOUBLE ) 
			return value_double;
		else
			throw new HopsException("Can not coerce an object of type " + get_valueType() + " into Double.");
	}
	
	public boolean getBooleanValue() throws HopsException {
		if ( get_valueType() == ValueType.BOOLEAN ) {
			return value_boolean;
		}
		else
			throw new HopsException("Can not coerce an object of type " + get_valueType() + " into Boolean.");
	}
	
	public String getStringValue() throws HopsException {
		if ( get_valueType() == ValueType.STRING ) {
			return value_string;
		}
		else
			throw new HopsException("Can not coerce an object of type " + get_valueType() + " into String.");
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		LiteralOp ret = new LiteralOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.value_double = value_double;
		ret.value_long = value_long;
		ret.value_string = value_string;
		ret.value_boolean = value_boolean;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		return false;
	}
}
