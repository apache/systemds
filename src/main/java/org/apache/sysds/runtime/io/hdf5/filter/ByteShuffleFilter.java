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

public class ByteShuffleFilter implements Filter {

	@Override
	public int getId() {
		return 2;
	}

	@Override
	public String getName() {
		return "shuffle";
	}

	@Override
	public byte[] decode(byte[] data, int[] filterData) {

		// Bytes in each element e.g float32 = 4 bytes
		final int dataSize = filterData[0];

		// A quick shortcut if no shuffling is needed
		if (dataSize == 1) {
			return data;
		}

		final int elements = data.length / dataSize;

		// shuffle doesn't change the size of the data it rearranges it
		final byte[] out = new byte[data.length];

		int pos = 0;
		for (int i = 0; i < dataSize; i++) {
			for (int j = 0; j < elements; j++) {
				out[j * dataSize + i] = data[pos];
				pos++; // step through the input array
			}
		}

		return out;
	}
}
