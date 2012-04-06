package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;

public class StringObject extends ScalarObject {
	private String _value;

	public StringObject (String val){
		this(null,val);
	}

	public StringObject(String name, String val){
		super(name, ValueType.STRING);
		_value = val;
	}

	public double getDoubleValue(){
		throw new UnsupportedOperationException();
	}

	public int getIntValue(){
		throw new UnsupportedOperationException();
	}

	public long getLongValue(){
		throw new UnsupportedOperationException();
	}
	
	public Object getValue(){
		return _value;
	}
	
	public String getStringValue(){
		return _value;
	}

	public boolean getBooleanValue(){
		throw new UnsupportedOperationException();
	}


}
