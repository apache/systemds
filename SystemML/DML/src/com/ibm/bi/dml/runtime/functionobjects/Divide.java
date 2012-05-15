package com.ibm.bi.dml.runtime.functionobjects;

public class Divide extends ValueFunction {

	private static Divide singleObj = null;
	
	private Divide() {
		// nothing to do here
	}
	
	public static Divide getDivideFnObject() {
		if ( singleObj == null )
			singleObj = new Divide();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return in1 / in2;
	}

	@Override
	public double execute(double in1, int in2) {
		return in1 / (double)in2;
	}

	@Override
	public double execute(int in1, double in2) {
		return (double)in1 / in2;
	}

	@Override
	public double execute(int in1, int in2) {
		return (double)in1 / (double)in2;
	}

}
