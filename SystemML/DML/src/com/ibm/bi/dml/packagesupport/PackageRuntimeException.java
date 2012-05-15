package com.ibm.bi.dml.packagesupport;

/**
 * Class to capture a runtime exception during package execution.
 * 
 * @author aghoting
 * 
 */
public class PackageRuntimeException extends RuntimeException {

	private static final long serialVersionUID = 7388224928778587925L;

	public PackageRuntimeException(String msg) {
		super(msg);
	}

	public PackageRuntimeException(Exception e) {
		super(e);
	}

	public PackageRuntimeException(String msg, Exception e) {
		super(msg,e);
	}

}
