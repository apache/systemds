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

package org.apache.sysml.hops;

import org.apache.sysml.lops.Data;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.util.UtilFunctions;


public class LiteralOp extends Hop 
{
	private double value_double = Double.NaN;
	private long value_long = Long.MAX_VALUE;
	private String value_string;
	private boolean value_boolean;

	// INT, DOUBLE, STRING, BOOLEAN

	private LiteralOp() {
		//default constructor for clone
	}
	
	public LiteralOp(double value) {
		super(String.valueOf(value), DataType.SCALAR, ValueType.DOUBLE);
		value_double = value;
	}

	public LiteralOp(long value) {
		super(String.valueOf(value), DataType.SCALAR, ValueType.INT);
		value_long = value;
	}

	public LiteralOp(String value) {
		super(value, DataType.SCALAR, ValueType.STRING);
		value_string = value;
	}

	public LiteralOp(boolean value) {
		super(String.valueOf(value), DataType.SCALAR, ValueType.BOOLEAN);
		value_boolean = value;
	}
	
	public LiteralOp(LiteralOp that) {
		super(that.getName(), that.getDataType(), that.getValueType());
		value_double = that.value_double;
		value_long = that.value_long;
		value_string = that.value_string;
		value_boolean = that.value_boolean;
	}

	@Override
	public void checkArity() throws HopsException {
		HopsException.check(_input.isEmpty(), this, "should have 0 inputs but has %d inputs", _input.size());
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
			Lop l = null;

			switch (getValueType()) {
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
			setLineNumbers(l);
			setLops(l);
		} 
		catch(LopsException e) {
			throw new HopsException(e);
		}
		
		//note: no reblock lop because always scalar
		
		return getLops();
	}

	@Override
	public String getOpString() {
		String val = null;
		switch (getValueType()) {
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
			default:
				val = "";
		}
		return "LiteralOp " + val;
	}
		
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		
		switch( getValueType() ) {
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
			default:
				ret = 0;
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
	
	public long getLongValue() 
	{
		switch( getValueType() ) {
			case INT:		
				return value_long;
			case DOUBLE:	
				return UtilFunctions.toLong(value_double);
			case STRING:
				return Long.parseLong(value_string);	
			case BOOLEAN: 
				return value_boolean ? 1 : 0;
			default:
				return -1;
		}
	}
	
	public double getDoubleValue() throws HopsException {
		switch( getValueType() ) {
			case INT:		
				return value_long;
			case DOUBLE:	
				return value_double;
			case STRING:
				return Double.parseDouble(value_string);
			default:
				throw new HopsException("Can not coerce an object of type " + getValueType() + " into Double.");
		}
	}
	
	public boolean getBooleanValue() throws HopsException {
		if ( getValueType() == ValueType.BOOLEAN ) {
			return value_boolean;
		}
		else
			throw new HopsException("Can not coerce an object of type " + getValueType() + " into Boolean.");
	}
	
	public String getStringValue() 
	{
		switch( getValueType() ) {
			case BOOLEAN:
				return String.valueOf(value_boolean);
			case INT:		
				return String.valueOf(value_long);
			case DOUBLE:	
				return String.valueOf(value_double);
			case STRING:
				return value_string;
			case OBJECT:
			case UNKNOWN:	
				//do nothing (return null)
		}
		
		return null;
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
