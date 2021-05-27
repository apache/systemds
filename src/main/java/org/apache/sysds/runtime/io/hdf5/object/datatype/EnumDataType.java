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
import org.apache.sysds.runtime.io.hdf5.dataset.EnumDatasetReader;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class for reading enum data type messages.
 *
 * @author James Mudd
 */
public class EnumDataType extends DataType {

    private final DataType baseType;
    private final Map<Integer, String> ordinalToName;

    public EnumDataType(ByteBuffer bb) {
        super(bb);

        baseType = DataType.readDataType(bb);

        int numberOfMembers = Utils.bitsToInt(classBits, 0, 16);

        List<String> names = new ArrayList<>();
        if(getVersion() == 3) { // v3 not padded
            for (int i = 0; i < numberOfMembers; i++) {
                names.add(Utils.readUntilNull(bb));
            }
        } else { // v1 or 2 are padded
            for (int i = 0; i < numberOfMembers; i++) {
                final int posBeforeName = bb.position();
                names.add(Utils.readUntilNull(bb));
                final int posAfterName = bb.position();
                final int bytesPastEight = (posAfterName - posBeforeName) % 8;
                if (bytesPastEight != 0) {
                    int bytesToSkip = 8 - bytesPastEight;
                    bb.position(bb.position() + bytesToSkip);
                }
            }
        }

        List<Integer> values = new ArrayList<>();
        for (int i = 0; i < numberOfMembers; i++) {
            values.add(Utils.readBytesAsUnsignedInt(bb, baseType.getSize()));
        }

        // now zip the 2 list into a map
        ordinalToName = new HashMap<>();
        for (int i = 0; i < numberOfMembers; i++) {
            ordinalToName.put(values.get(i), names.get(i));
        }

    }

    @Override
    public Class<?> getJavaType() {
        return String.class;
    }

    public DataType getBaseType() {
        return baseType;
    }

    public Map<Integer, String> getEnumMapping() {
        return ordinalToName;
    }

    @Override
    public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
        return EnumDatasetReader.readEnumDataset(this, buffer, dimensions);
    }
}
