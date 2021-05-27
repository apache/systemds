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
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfTypeException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.apache.sysds.runtime.io.hdf5.Utils.readBytesAsUnsignedLong;
import static org.apache.sysds.runtime.io.hdf5.Utils.stripLeadingIndex;

/**
 * Class to represent a Reference data type.
 *
 * <p>
 * <a href=
 * "https://support.hdfgroup.org/HDF5/doc/H5.format.html#DatatypeMessage">Datatypes</a>
 * </p>
 *
 * @author Gisa Meier
 */
public class Reference extends DataType {

	public enum ReferenceType{

		/** Object Reference: A reference to another object in the HDF5 file.*/
		OBJECT,

		/** Dataset Region Reference: A reference to a region within a dataset in the HDF5 file. */
		REGION
	}

	private final ReferenceType type;

	public Reference(ByteBuffer bb) {
		super(bb);

		int rawType = Utils.bitsToInt(classBits, 0, 4);
		switch (rawType) {
			case 0:
				type = ReferenceType.OBJECT;
				break;
			case 1:
				type = ReferenceType.REGION;
				throw new UnsupportedHdfException("Dataset Region references not supported for Reference data type");
			default:
				throw new HdfException("Unrecognized Reference type. Index is: " + rawType);
		}

	}

	public ReferenceType getType() {
		return type;
	}

	@Override
	public Class<?> getJavaType() {
		//object reference type handles addresses, which are always longs
		if (getSize() > 8) {
			throw new HdfTypeException("Unsupported address size in reference data type " + getSize() + "bytes");
		}
		return long.class;
	}

	@Override
	public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		final Object data = Array.newInstance(getJavaType(), dimensions);
		fillData(data, dimensions, buffer.order(ByteOrder.LITTLE_ENDIAN));
		return data;
	}

	private void fillData(Object data, int[] dims, ByteBuffer buffer) {
		if (dims.length > 1) {
			for (int i = 0; i < dims[0]; i++) {
				Object newArray = Array.get(data, i);
				fillData(newArray, stripLeadingIndex(dims), buffer);
			}
		} else {
			for (int i = 0; i < Array.getLength(data); i++) {
				Array.set(data, i, readBytesAsUnsignedLong(buffer, getSize()));
			}
		}
	}

}
