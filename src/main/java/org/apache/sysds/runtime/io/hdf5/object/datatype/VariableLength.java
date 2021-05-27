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
import org.apache.sysds.runtime.io.hdf5.dataset.VariableLengthDatasetReader;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.object.datatype.StringData.PaddingType;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class VariableLength extends DataType {

	private final int type;
	private final PaddingType paddingType;
	private final Charset encoding;
	private final DataType parent;

	public VariableLength(ByteBuffer bb) {
		super(bb);

		type = Utils.bitsToInt(classBits, 0, 4);

		final int paddingTypeValue = Utils.bitsToInt(classBits, 4, 4);
		switch (paddingTypeValue) {
			case 0:
				paddingType = PaddingType.NULL_TERMINATED;
				break;
			case 1:
				paddingType = PaddingType.NULL_PADDED;
				break;
			case 2:
				paddingType = PaddingType.SPACE_PADDED;
				break;
			default:
				throw new HdfException("Unrecognized padding type. Value is: " + paddingTypeValue);
		}

		final int characterEncoding = Utils.bitsToInt(classBits, 8, 4);
		switch (characterEncoding) {
		case 0:
			encoding = StandardCharsets.US_ASCII;
			break;
		case 1:
			encoding = StandardCharsets.UTF_8;
			break;
		default:
			throw new HdfException("Unrecognized character encoding = " + characterEncoding);
		}

		parent = DataType.readDataType(bb);
	}

	public boolean isVariableLengthString() {
		return type == 1;
	}

	public int getType() {
		return type;
	}

	public PaddingType getPaddingType() {
		if(!isVariableLengthString()) {
			throw new HdfException("Cannot get padding type for variable length dataset thats not string type");
		}
		return paddingType;
	}

	public Charset getEncoding() {
		if(!isVariableLengthString()) {
			throw new HdfException("Cannot get encoding for variable length dataset thats not string type");
		}
		return encoding;
	}

	public DataType getParent() {
		return parent;
	}

	@Override
	public Class<?> getJavaType() {
		if (isVariableLengthString()) {
			return String.class;
		} else {
			return Object.class;
		}
	}

	@Override
	public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		return  VariableLengthDatasetReader.readDataset(this, buffer, dimensions, hdfFc);
	}
}
