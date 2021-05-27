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

import com.ning.compress.lzf.LZFException;
import com.ning.compress.lzf.util.ChunkDecoderFactory;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfFilterException;

public class LzfFilter implements Filter {

	/**
	 * Id defined in https://support.hdfgroup.org/services/filters.html
	 *
	 * @return Defined value, 32000
	 */
	@Override
	public int getId() {
		return 32000;
	}

	/**
	 * The name of this filter, "lzf
	 *
	 * @return "lzf"
	 */
	@Override
	public String getName() {
		return "lzf";
	}

	@Override
	public byte[] decode(byte[] encodedData, int[] filterData) {
		final int compressedLength = encodedData.length;
		final int uncompressedLength = filterData[2];

		if (compressedLength == uncompressedLength) {
			return encodedData;
		}

		final byte[] output = new byte[uncompressedLength];

		try {
			ChunkDecoderFactory.safeInstance().decodeChunk(encodedData, 0, output, 0, uncompressedLength);
		} catch (final LZFException e) {
			throw new HdfFilterException("Inflating failed", e);
		}
		return output;
	}
}
