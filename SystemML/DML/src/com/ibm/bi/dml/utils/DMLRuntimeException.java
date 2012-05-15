package com.ibm.bi.dml.utils;

public class DMLRuntimeException extends DMLException {

	/**
	 * This exception should be thrown to flag runtime errors -- DML equivalent to java.lang.RuntimeException.
	 */
	private static final long serialVersionUID = 1L;

	public DMLRuntimeException(String string) {
		super(string);
	}
	
	public DMLRuntimeException(Exception e) {
		super(e);
	}

	public DMLRuntimeException(String string, Exception ex){
		super(string,ex);
	}
}
