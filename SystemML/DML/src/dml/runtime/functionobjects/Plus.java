package dml.runtime.functionobjects;

import dml.utils.DMLRuntimeException;

// Singleton class

public class Plus extends ValueFunction {

	private static Plus singleObj = null;
	
	private Plus() {
		// nothing to do here
	}
	
	public static Plus getPlusFnObject() {
		if ( singleObj == null )
			singleObj = new Plus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 + in2;
	}

	@Override
	public double execute(double in1, int in2) {
		return in1 + in2;
	}

	@Override
	public double execute(int in1, double in2) {
		return in1 + in2;
	}

	@Override
	public double execute(int in1, int in2) {
		return in1 + in2;
	}

	public String execute ( String in1, String in2 ) throws DMLRuntimeException {
		return in1 + in2;
	}
	
}
