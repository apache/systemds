package com.ibm.bi.dml.runtime.functionobjects;

public class Not extends ValueFunction {
	public static Not singleObj = null;

	private Not() {
		// nothing to do here
	}
	
	public static Not getNotFnObject() {
		if ( singleObj == null )
			singleObj = new Not();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public boolean execute(boolean in) {
		return !in;
	}
}
