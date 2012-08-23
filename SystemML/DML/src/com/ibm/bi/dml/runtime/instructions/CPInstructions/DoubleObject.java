package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class DoubleObject extends ScalarObject {

	private double _value;

	public DoubleObject(double val){
		this(null,val);
	}

	public DoubleObject(String name, double val){
		super(name, ValueType.DOUBLE);
		_value = val;
	}

	public double getDoubleValue(){
		return _value;
	}

	public int getIntValue(){
		return UtilFunctions.toInt(_value);
	}
	
	public long getLongValue() {
		return UtilFunctions.toLong(_value);
	}
	

	public Object getValue(){
		return _value;
	}
	
	public String getStringValue(){
		return Double.toString(_value);
	}

	public boolean getBooleanValue(){
		throw new UnsupportedOperationException();
	}

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}

	

}
