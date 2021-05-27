/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.filter;

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfFilterException;

/**
 * Interface to be implemented to be a HDF5 filter.
 *
 * @author James Mudd
 */
public interface Filter {

	/**
	 * Gets the ID of this filter, this must match the ID in the dataset header.
	 *
	 * @return the ID of this filter
	 */
	int getId();

	/**
	 * Gets the name of this filter e.g. 'deflate', 'shuffle'
	 *
	 * @return the name of this filter
	 */
	String getName();

	/**
	 * Applies this filter to decode data. If the decode fails a
	 * {@link HdfFilterException} will be thrown. This method must be thread safe,
	 * multiple thread may use the filter simultaneously.
	 *
	 * @param encodedData the data to be decoded
	 * @param filterData  the settings from the file this filter was used with. e.g.
	 *                    compression level.
	 * @return the decoded data
	 * @throws HdfFilterException if the decode operation fails
	 */
	byte[] decode(byte[] encodedData, int[] filterData);

}
