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

public class HdfException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public HdfException(String message) {
		super(message);
	}

	public HdfException(String message, Throwable throwable) {
		super(message, throwable);
	}
}
