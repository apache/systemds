/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.exceptions;

/**
 * Exception to be thrown when operations involving filters fail.
 *
 * @author James Mudd
 */
public class HdfFilterException extends HdfException {

	private static final long serialVersionUID = 1L;

	public HdfFilterException(String message) {
		super(message);
	}

	public HdfFilterException(String message, Throwable throwable) {
		super(message, throwable);
	}

}
