package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;

public class IntObject extends ScalarObject  {

	private int _value;

	public IntObject(int val){
		this(null,val);
	}

	public IntObject(String name,int val){
		super(name, ValueType.INT);
		_value = val;
	}

	public int getIntValue(){
		return _value;
	}

	public double getDoubleValue(){
		return (double) _value;
	}

	public long getLongValue(){
		return (long) _value;
	}
	
	public Object getValue(){
		return _value;
	}

	public boolean getBooleanValue(){
		throw new UnsupportedOperationException();
	}

	public String getStringValue(){
		return Integer.toString(_value);
	}

}
