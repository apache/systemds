package com.ibm.bi.dml.runtime.functionobjects;

public class Equals extends ValueFunction {

	public static Equals singleObj = null;

	private Equals() {
		// nothing to do here
	}
	
	public static Equals getEqualsFnObject() {
		if ( singleObj == null )
			singleObj = new Equals();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (Double.compare(in1, in2) == 0 ? 1.0 : 0.0);
	}
	
	@Override
	public boolean compare(boolean in1, boolean in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(double in1, double in2) {
		return (Double.compare(in1, in2) == 0);
	}

	@Override
	public boolean compare(int in1, int in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(double in1, int in2) {
		return (Double.compare(in1, in2) == 0);
	}

	@Override
	public boolean compare(int in1, double in2) {
		return (Double.compare(in1, in2) == 0);
	}
}
