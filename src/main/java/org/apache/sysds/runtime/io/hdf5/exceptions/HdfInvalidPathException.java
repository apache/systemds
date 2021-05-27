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

import java.io.File;

/**
 * Thrown when a path inside a HDF5 file is invalid. It may contain invalid
 * characters or not be found in the file.
 *
 * @author James Mudd
 */
public class HdfInvalidPathException extends HdfException {

	private static final long serialVersionUID = 1L;

	private final String path;
	private final File file;

	public HdfInvalidPathException(String path, File file) {
		super("The path '" + path + "' could not be found in the HDF5 file '" + file.getAbsolutePath() + "'");
		this.path = path;
		this.file = file;
	}

	public String getPath() {
		return path;
	}

	public File getFile() {
		return file;
	}

}
