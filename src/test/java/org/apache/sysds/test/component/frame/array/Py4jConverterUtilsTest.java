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

 package org.apache.sysds.test.component.frame.array;

 
 import static org.junit.Assert.assertEquals;
 import static org.junit.Assert.assertNotNull;
 
 import java.nio.ByteBuffer;
 import java.nio.ByteOrder;
 import java.nio.charset.StandardCharsets;
 
 import org.apache.sysds.common.Types;


 import org.apache.sysds.runtime.util.Py4jConverterUtils;
 import org.apache.sysds.runtime.frame.data.columns.Array;
 import org.junit.Test;

 
 public class Py4jConverterUtilsTest {
 
	@Test
    public void testConvertUINT8() {
        int numElements = 4;
        byte[] data = {1, 2, 3, 4};
        Array<?> result = Py4jConverterUtils.convert(data, numElements, Types.ValueType.UINT8);
        assertNotNull(result);
        assertEquals(4, result.size());
        assertEquals(1, result.get(0));
        assertEquals(2, result.get(1));
        assertEquals(3, result.get(2));
        assertEquals(4, result.get(3));
    }

    @Test
    public void testConvertINT32() {
        int numElements = 4;
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * numElements);
        buffer.order(ByteOrder.nativeOrder());
        for (int i = 1; i <= numElements; i++) {
            buffer.putInt(i);
        }
        Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.INT32);
        assertNotNull(result);
        assertEquals(4, result.size());
        assertEquals(1, result.get(0));
        assertEquals(2, result.get(1));
        assertEquals(3, result.get(2));
        assertEquals(4, result.get(3));
    }

    @Test
    public void testConvertFP64() {
        int numElements = 4;
        ByteBuffer buffer = ByteBuffer.allocate(Double.BYTES * numElements);
        buffer.order(ByteOrder.nativeOrder());
        for (double i = 1.1; i <= numElements + 1; i += 1.0) {
            buffer.putDouble(i);
        }
        Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.FP64);
        assertNotNull(result);
        assertEquals(4, result.size());
        assertEquals(1.1, result.get(0));
		assertEquals(2.1, result.get(1));
		assertEquals(3.1, result.get(2));
		assertEquals(4.1, result.get(3));
    }

    @Test
    public void testConvertBoolean() {
        int numElements = 4;
        byte[] data = {1, 0, 1, 0};
        Array<?> result = Py4jConverterUtils.convert(data, numElements, Types.ValueType.BOOLEAN);
        assertNotNull(result);
        assertEquals(4, result.size());
        assertEquals(true, result.get(0));
        assertEquals(false, result.get(1));
        assertEquals(true, result.get(2));
        assertEquals(false, result.get(3));
    }

    @Test
    public void testConvertString() {
        int numElements = 2;
        String[] strings = {"hello", "world"};
        ByteBuffer buffer = ByteBuffer.allocate(4 + strings[0].length() + 4 + strings[1].length());
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (String s : strings) {
            buffer.order(ByteOrder.BIG_ENDIAN);
            buffer.putInt(s.length());
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.put(s.getBytes(StandardCharsets.UTF_8));
        }
        Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.STRING);
        assertNotNull(result);
        assertEquals(2, result.size());
        assertEquals("hello", result.get(0));
        assertEquals("world", result.get(1));
    }
}
