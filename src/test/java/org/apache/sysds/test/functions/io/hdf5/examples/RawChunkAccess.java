/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.test.functions.io.hdf5.examples;

import org.apache.sysds.runtime.io.hdf5.HDF5File;
import org.apache.sysds.runtime.io.hdf5.api.Dataset;
import org.apache.sysds.runtime.io.hdf5.api.ChunkedDataset;
import org.apache.commons.lang3.ArrayUtils;

import java.io.File;
import java.nio.ByteBuffer;

/**
 * Example application for raw chunk access from HDF5
 *
 * @author James Mudd
 */
public class RawChunkAccess {
	public static void main(String[] args) {
		File file = new File(args[0]);

		try (HDF5File HDF5File = new HDF5File(file)) {
			Dataset dataset = HDF5File.getDatasetByPath(args[1]);
			if (dataset instanceof ChunkedDataset) {
				ChunkedDataset chunkedDataset = (ChunkedDataset) dataset;
				int[] chunkOffset = new int[chunkedDataset.getChunkDimensions().length];
				System.out.println("Chunk offset: " + ArrayUtils.toString(chunkOffset));
				// For the example just get the zero chunk but you can get any
				ByteBuffer rawChunkBuffer = chunkedDataset.getRawChunkBuffer(chunkOffset);
				// If you need the buffer just use it directly here, if you want the byte[]
				byte[] byteArray = new byte[rawChunkBuffer.capacity()];
				rawChunkBuffer.get(byteArray);
				// Now you have the byte[] to use as you like
				System.out.println("Raw bytes: " + ArrayUtils.toString(byteArray));
			} else {
				throw new IllegalArgumentException("Dataset is not chunked");
			}
		}
	}
}
