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
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.apache.sysds.runtime.io.hdf5.Utils.stripLeadingIndex;

public class BitField extends DataType implements OrderedDataType {
    private final ByteOrder order;
    private final boolean lowPadding;
    private final boolean highPadding;
    private final short bitOffset;
    private final short bitPrecision;

    public BitField(ByteBuffer bb) {
        super(bb);

        if (classBits.get(0)) {
            order = ByteOrder.BIG_ENDIAN;
        } else {
            order = ByteOrder.LITTLE_ENDIAN;
        }

        lowPadding = classBits.get(1);
        highPadding = classBits.get(2);

        bitOffset = bb.getShort();
        bitPrecision = bb.getShort();
    }

    @Override
    public ByteOrder getByteOrder() {
        return order;
    }

    public boolean isLowPadding() {
        return lowPadding;
    }

    public boolean isHighPadding() {
        return highPadding;
    }

    public short getBitOffset() {
        return bitOffset;
    }

    public short getBitPrecision() {
        return bitPrecision;
    }

    @Override
    public Class<?> getJavaType() {
        return boolean.class;
    }

    @Override
    public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
        final Object data = Array.newInstance(getJavaType(), dimensions);
        fillBitfieldData(data, dimensions, buffer.order(getByteOrder()));
        return data;
    }

    private static void fillBitfieldData(Object data, int[] dims, ByteBuffer buffer) {
        if (dims.length > 1) {
            for (int i = 0; i < dims[0]; i++) {
                Object newArray = Array.get(data, i);
                fillBitfieldData(newArray, stripLeadingIndex(dims), buffer);
            }
        } else {
            for (int i = 0; i < Array.getLength(data); i++) {
                Array.set(data, i, buffer.get() == 1);
            }
        }
    }


}
