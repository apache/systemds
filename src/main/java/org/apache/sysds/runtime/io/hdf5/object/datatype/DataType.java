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

import org.apache.sysds.runtime.io.hdf5.BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public abstract class DataType {

	private  int version;
	private  int dataClass;
	private  int size; // In bytes
	protected BitSet classBits;

	public DataType() {}

	public static DataType readDataType(ByteBuffer bb) {
		// Mark buffer position
		bb.mark();

		// Class and version
		final BitSet classAndVersion = BitSet.valueOf(new byte[] {bb.get()});
		int version = Utils.bitsToInt(classAndVersion, 4, 4);
		int dataClass = Utils.bitsToInt(classAndVersion, 0, 4);

		if(version == 0 || version > 3) {
			throw new HdfException("Unrecognized datatype version '" + version + "' detected");
		}

		// Move the buffer back to the start of the data type message
		bb.reset();

		switch(dataClass) {
			case 0: // Fixed point
				return new FixedPoint(bb);
			case 1: // Floating point
				return new FloatingPoint(bb);
			case 2: // Time
				throw new UnsupportedHdfException("Time data type is not yet supported");
			case 3: // String
				return new StringData(bb);
			case 4: // Bit field
				return new BitField(bb);
			case 5: // Opaque
				throw new UnsupportedHdfException("Opaque data type is not yet supported");
			case 6: // Compound
				return new CompoundDataType(bb);
			case 7: // Reference
				return new Reference(bb);
			case 8: // Enum
				return new EnumDataType(bb);
			case 9: // Variable length
				return new VariableLength(bb);
			case 10: // Array
				return new ArrayDataType(bb);
			default:
				throw new HdfException("Unrecognized data class = " + dataClass);
		}

	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}

	public BufferBuilder toBuffer(BufferBuilder header) {
		return header;
	}



	protected DataType(ByteBuffer bb) {

		// Class and version
		final BitSet classAndVersion = BitSet.valueOf(new byte[] {bb.get()});
		dataClass = Utils.bitsToInt(classAndVersion, 0, 4);
		version = Utils.bitsToInt(classAndVersion, 4, 4);

		byte[] classBytes = new byte[3];
		bb.get(classBytes);
		classBits = BitSet.valueOf(classBytes);
		// Size
		size = Utils.readBytesAsUnsignedInt(bb, 4);

	}

	public int getVersion() {
		return version;
	}

	public int getDataClass() {
		return dataClass;
	}

	public int getSize() {
		return size;
	}

	public abstract Class<?> getJavaType();

	public abstract Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc);

}
