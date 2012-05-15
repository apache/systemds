package com.ibm.bi.dml.runtime.functionobjects;

public class EqualsReturnDouble extends ValueFunction {

	public static EqualsReturnDouble singleObj = null;

	private EqualsReturnDouble() {
		// nothing to do here
	}
	
	public static EqualsReturnDouble getEqualsReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new EqualsReturnDouble();
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

}
