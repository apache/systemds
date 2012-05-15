package com.ibm.bi.dml.runtime.functionobjects;

public class LessThanEqualsReturnDouble extends ValueFunction {

	public static LessThanEqualsReturnDouble singleObj = null;

	private LessThanEqualsReturnDouble() {
		// nothing to do here
	}
	
	public static LessThanEqualsReturnDouble getLessThanEqualsReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new LessThanEqualsReturnDouble();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (Double.compare(in1, in2) <= 0 ? 1.0 : 0.0);
	}
}
