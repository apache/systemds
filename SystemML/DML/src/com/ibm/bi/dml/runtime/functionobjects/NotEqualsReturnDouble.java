package com.ibm.bi.dml.runtime.functionobjects;

public class NotEqualsReturnDouble extends ValueFunction {
	
	public static NotEqualsReturnDouble singleObj = null;

	private NotEqualsReturnDouble() {
		// nothing to do here
	}
	
	public static NotEqualsReturnDouble getNotEqualsReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new NotEqualsReturnDouble();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (in1 != in2 ? 1.0 : 0.0);
	}
}
