/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysds.runtime.io.hdf5.dataset;

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfTypeException;
import org.apache.sysds.runtime.io.hdf5.object.datatype.DataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.EnumDataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.FixedPoint;

import java.lang.reflect.Array;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Map;

import static org.apache.sysds.runtime.io.hdf5.Utils.stripLeadingIndex;

public class EnumDatasetReader {

    /** No instances */
    private EnumDatasetReader() {
        throw new AssertionError("No instances of EnumDatasetReader");
    }

    public static Object readEnumDataset(EnumDataType enumDataType, ByteBuffer buffer, int[] dimensions) {

        final DataType baseType = enumDataType.getBaseType();
        if (baseType instanceof FixedPoint) {
            Object data = Array.newInstance(String.class, dimensions);

            FixedPoint fixedPoint = (FixedPoint) baseType;

            switch (fixedPoint.getSize()) {
                case 1:
                    fillDataUnsigned(data, dimensions, buffer.order(fixedPoint.getByteOrder()), enumDataType.getEnumMapping());
                    break;
                case 2:
                    fillDataUnsigned(data, dimensions, buffer.order(fixedPoint.getByteOrder()).asShortBuffer(), enumDataType.getEnumMapping());
                    break;
                case 4:
                    fillDataUnsigned(data, dimensions, buffer.order(fixedPoint.getByteOrder()).asIntBuffer(), enumDataType.getEnumMapping());
                    break;
                case 8:
                    fillDataUnsigned(data, dimensions, buffer.order(fixedPoint.getByteOrder()).asLongBuffer(), enumDataType.getEnumMapping());
                    break;
                default:
                    throw new HdfTypeException(
                            "Unsupported signed integer type size " + fixedPoint.getSize() + " bytes");
            }

            return data;
        } else {
            throw new HdfException("Trying to fill enum dataset with non-integer base type: " + baseType);
        }
    }

    private static void fillDataUnsigned(Object data, int[] dims, ByteBuffer buffer, Map<Integer, String> enumMapping) {
        if (dims.length > 1) {
            for (int i = 0; i < dims[0]; i++) {
                Object newArray = Array.get(data, i);
                fillDataUnsigned(newArray, stripLeadingIndex(dims), buffer, enumMapping);
            }
        } else {
            final byte[] tempBuffer = new byte[dims[dims.length - 1]];
            buffer.get(tempBuffer);
            // Convert to enum values
            String[] stringData = (String[]) data;
            for (int i = 0; i < tempBuffer.length; i++) {
                stringData[i] = enumMapping.get(Byte.toUnsignedInt(tempBuffer[i]));
            }
        }
    }

    private static void fillDataUnsigned(Object data, int[] dims, ShortBuffer buffer, Map<Integer, String> enumMapping) {
        if (dims.length > 1) {
            for (int i = 0; i < dims[0]; i++) {
                Object newArray = Array.get(data, i);
                fillDataUnsigned(newArray, stripLeadingIndex(dims), buffer, enumMapping);
            }
        } else {
            final short[] tempBuffer = new short[dims[dims.length - 1]];
            buffer.get(tempBuffer);
            // Convert to enum values
            String[] stringData = (String[]) data;
            for (int i = 0; i < tempBuffer.length; i++) {
                stringData[i] = enumMapping.get(Short.toUnsignedInt(tempBuffer[i]));
            }
        }
    }

    private static void fillDataUnsigned(Object data, int[] dims, IntBuffer buffer, Map<Integer, String> enumMapping) {
        if (dims.length > 1) {
            for (int i = 0; i < dims[0]; i++) {
                Object newArray = Array.get(data, i);
                fillDataUnsigned(newArray, stripLeadingIndex(dims), buffer, enumMapping);
            }
        } else {
            final int[] tempBuffer = new int[dims[dims.length - 1]];
            buffer.get(tempBuffer);
            // Convert to enum values
            String[] stringData = (String[]) data;
            for (int i = 0; i < tempBuffer.length; i++) {
                stringData[i] = enumMapping.get(Math.toIntExact(Integer.toUnsignedLong(tempBuffer[i])));
            }
        }
    }

    private static void fillDataUnsigned(Object data, int[] dims, LongBuffer buffer, Map<Integer, String> enumMapping) {
        if (dims.length > 1) {
            for (int i = 0; i < dims[0]; i++) {
                Object newArray = Array.get(data, i);
                fillDataUnsigned(newArray, stripLeadingIndex(dims), buffer, enumMapping);
            }
        } else {
            final long[] tempBuffer = new long[dims[dims.length - 1]];
            final ByteBuffer tempByteBuffer = ByteBuffer.allocate(8);
            buffer.get(tempBuffer);
            // Convert to enum values
            String[] stringData = (String[]) data;
            for (int i = 0; i < tempBuffer.length; i++) {
                tempByteBuffer.putLong(0, tempBuffer[i]);
                stringData[i] = enumMapping.get((new BigInteger(1, tempByteBuffer.array())).intValueExact());
            }
        }
    }

}
