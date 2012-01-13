package dml.runtime.instructions.CPInstructions;

public class DoubleObject extends ScalarObject {

	private double _value;

	public DoubleObject(double val){
		this(null,val);
	}

	public DoubleObject(String name, double val){
		super(name);
		_value = val;
	}

	public double getDoubleValue(){
		return _value;
	}

	public int getIntValue(){
		return (int)_value;
	}
	
	public long getLongValue() {
		return (long)_value;
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

	

}
