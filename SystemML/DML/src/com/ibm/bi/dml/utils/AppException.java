package com.ibm.bi.dml.utils;

public class AppException extends DMLException {

	/**
	 * This exception should be thrown to flag errors by systemML apps
	 * including data transformation, post-processing etc.
	 */
	private static final long serialVersionUID = 1L;

	public AppException(String string) {
		super(string);
	}

	public AppException(String string, Exception ex){
		super(string,ex);
	}
}