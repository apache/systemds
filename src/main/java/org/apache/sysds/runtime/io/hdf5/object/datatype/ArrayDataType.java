/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.object.datatype;

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.dataset.DatasetReader;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.commons.lang3.ArrayUtils;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;

import static org.apache.sysds.runtime.io.hdf5.Utils.readBytesAsUnsignedInt;

/**
 * Class for reading array data type messages.
 */
public class ArrayDataType extends DataType {

	private final DataType baseType;
	private final int[] dimensions;

	public ArrayDataType(ByteBuffer bb) {
		super(bb);

		final int dimensionsSize = readBytesAsUnsignedInt(bb, 1);
		dimensions = new int[dimensionsSize];

		if(getVersion() == 2) {
			// Skip 3 bytes
			bb.position(bb.position() + 3);
		}

		for(int i = 0; i < dimensions.length; i++) {
			dimensions[i] = readBytesAsUnsignedInt(bb, 4);

			if(getVersion() == 2) {
				// Skip Permutation Index not supported anyway
				bb.position(bb.position() + 4);
			}
		}

		baseType = DataType.readDataType(bb);
	}

	@Override public Class<?> getJavaType() {
		return Array.newInstance(baseType.getJavaType(), 0).getClass();
	}

	public DataType getBaseType() {
		return baseType;
	}

	public int[] getDimensions() {
		return ArrayUtils.clone(dimensions);
	}

	@Override public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		if(dimensions.length != 1) {
			throw new HdfException("Multi dimension array data types are not supported");
		}
		final Object data = Array.newInstance(getJavaType(), dimensions);
		for(int i = 0; i < dimensions[0]; i++) {
			final ByteBuffer elementBuffer = Utils
				.createSubBuffer(buffer, getBaseType().getSize() * getDimensions()[0]);
			final Object elementDataset = DatasetReader
				.readDataset(getBaseType(), elementBuffer, getDimensions(), hdfFc);
			Array.set(data, i, elementDataset);
		}
		return data;
	}
}
