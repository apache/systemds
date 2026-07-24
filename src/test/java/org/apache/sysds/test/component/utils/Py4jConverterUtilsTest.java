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

package org.apache.sysds.test.component.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.Py4jConverterUtils;
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
		for(int i = 1; i <= numElements; i++) {
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
	public void testConvertINT64() {
		int numElements = 4;
		ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES * numElements);
		buffer.order(ByteOrder.nativeOrder());
		for(int i = 1; i <= numElements; i++) {
			buffer.putLong((long) i);
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.INT64);
		assertNotNull(result);
		assertEquals(4, result.size());
		assertEquals(1L, result.get(0));
		assertEquals(2L, result.get(1));
		assertEquals(3L, result.get(2));
		assertEquals(4L, result.get(3));
	}


	@Test
	public void testConvertHASH32() {
		int numElements = 4;
		ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * numElements);
		buffer.order(ByteOrder.nativeOrder());
		for(int i = 1; i <= numElements; i++) {
			buffer.putInt(i);
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.HASH32);
		assertNotNull(result);
		assertEquals(4, result.size());
		assertEquals("1", result.get(0));
		assertEquals("2", result.get(1));
		assertEquals("3", result.get(2));
		assertEquals("4", result.get(3));
	}

	@Test
	public void testConvertHASH64() {
		int numElements = 4;
		ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES * numElements);
		buffer.order(ByteOrder.nativeOrder());
		for(int i = 1; i <= numElements; i++) {
			buffer.putLong((long) i);
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.HASH64);
		assertNotNull(result);
		assertEquals(4, result.size());
		assertEquals("1", result.get(0));
		assertEquals("2", result.get(1));
		assertEquals("3", result.get(2));
		assertEquals("4", result.get(3));
	}

	@Test
	public void testConvertFP32() {
		int numElements = 4;
		ByteBuffer buffer = ByteBuffer.allocate(Float.BYTES * numElements);
		buffer.order(ByteOrder.nativeOrder());
		for(float i = 1.1f; i <= numElements + 1; i += 1.0) {
			buffer.putFloat(i);
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.FP32);
		assertNotNull(result);
		assertEquals(4, result.size());
		assertEquals(1.1f, result.get(0));
		assertEquals(2.1f, result.get(1));
		assertEquals(3.1f, result.get(2));
		assertEquals(4.1f, result.get(3));
	}

	@Test
	public void testConvertFP64() {
		int numElements = 4;
		ByteBuffer buffer = ByteBuffer.allocate(Double.BYTES * numElements);
		buffer.order(ByteOrder.nativeOrder());
		for(double i = 1.1; i <= numElements + 1; i += 1.0) {
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
		for(String s : strings) {
			buffer.putInt(s.length());
			buffer.put(s.getBytes(StandardCharsets.UTF_8));
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), numElements, Types.ValueType.STRING);
		assertNotNull(result);
		assertEquals(2, result.size());
		assertEquals("hello", result.get(0));
		assertEquals("world", result.get(1));
	}

	@Test
	public void testConvertChar() {
		char[] c = {'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'};
		ByteBuffer buffer = ByteBuffer.allocate(Character.BYTES * c.length);
		buffer.order(ByteOrder.LITTLE_ENDIAN);
		for(char s : c) {
			buffer.putChar(s);
		}
		Array<?> result = Py4jConverterUtils.convert(buffer.array(), c.length, Types.ValueType.CHARACTER);
		assertNotNull(result);
		assertEquals(c.length, result.size());

		for(int i = 0; i < c.length; i++) {
			assertEquals(c[i], result.get(i));
		}
	}

	@Test
	public void testConvertRow() {
		int numElements = 4;
		byte[] data = {1, 2, 3, 4};
		Object[] row = Py4jConverterUtils.convertRow(data, numElements, Types.ValueType.UINT8);
		assertNotNull(row);
		assertEquals(4, row.length);
		assertEquals(1, row[0]);
		assertEquals(2, row[1]);
		assertEquals(3, row[2]);
		assertEquals(4, row[3]);
	}

	@Test
	public void testConvertFused() {
		int numElements = 1;
		byte[] data = {1, 2, 3, 4};
		Types.ValueType[] valueTypes = {ValueType.UINT8, ValueType.UINT8, ValueType.UINT8, ValueType.UINT8};
		Array<?>[] arrays = Py4jConverterUtils.convertFused(data, numElements, valueTypes);
		assertNotNull(arrays);
		assertEquals(4, arrays.length);
		for(int i = 0; i < 4; i++) {
			assertEquals(1 + i, arrays[i].get(0));
		}
	}

	@Test(expected = Exception.class)
	public void nullData() {
		Py4jConverterUtils.convert(null, 14, ValueType.BOOLEAN);
	}

	@Test(expected = Exception.class)
	public void nullValueType() {
		Py4jConverterUtils.convert(new byte[] {1, 2, 3}, 14, null);
	}

	@Test(expected = Exception.class)
	public void unknownValueType() {
		Py4jConverterUtils.convert(new byte[] {1, 2, 3}, 14, ValueType.UNKNOWN);
	}

	@Test
	public void testConvertPy4JArrayToMBFP64() {
		int rlen = 2;
		int clen = 3;
		ByteBuffer buffer = ByteBuffer.allocate(Double.BYTES * rlen * clen);
		buffer.order(ByteOrder.nativeOrder());
		double[] values = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
		for(double val : values) {
			buffer.putDouble(val);
		}
		MatrixBlock mb = Py4jConverterUtils.convertPy4JArrayToMB(buffer.array(), rlen, clen);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertEquals(1.1, mb.get(0, 0), 0.0001);
		assertEquals(2.2, mb.get(0, 1), 0.0001);
		assertEquals(3.3, mb.get(0, 2), 0.0001);
		assertEquals(4.4, mb.get(1, 0), 0.0001);
		assertEquals(5.5, mb.get(1, 1), 0.0001);
		assertEquals(6.6, mb.get(1, 2), 0.0001);
	}

	@Test
	public void testConvertPy4JArrayToMBUINT8() {
		int rlen = 2;
		int clen = 2;
		byte[] data = {(byte) 1, (byte) 2, (byte) 3, (byte) 4};
		MatrixBlock mb = Py4jConverterUtils.convertPy4JArrayToMB(data, rlen, clen, ValueType.UINT8);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertEquals(1.0, mb.get(0, 0), 0.0001);
		assertEquals(2.0, mb.get(0, 1), 0.0001);
		assertEquals(3.0, mb.get(1, 0), 0.0001);
		assertEquals(4.0, mb.get(1, 1), 0.0001);
	}

	@Test
	public void testConvertPy4JArrayToMBINT32() {
		int rlen = 2;
		int clen = 2;
		ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES * rlen * clen);
		buffer.order(ByteOrder.nativeOrder());
		int[] values = {10, 20, 30, 40};
		for(int val : values) {
			buffer.putInt(val);
		}
		MatrixBlock mb = Py4jConverterUtils.convertPy4JArrayToMB(buffer.array(), rlen, clen, ValueType.INT32);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertEquals(10.0, mb.get(0, 0), 0.0001);
		assertEquals(20.0, mb.get(0, 1), 0.0001);
		assertEquals(30.0, mb.get(1, 0), 0.0001);
		assertEquals(40.0, mb.get(1, 1), 0.0001);
	}

	@Test
	public void testConvertPy4JArrayToMBFP32() {
		int rlen = 2;
		int clen = 2;
		ByteBuffer buffer = ByteBuffer.allocate(Float.BYTES * rlen * clen);
		buffer.order(ByteOrder.nativeOrder());
		float[] values = {1.5f, 2.5f, 3.5f, 4.5f};
		for(float val : values) {
			buffer.putFloat(val);
		}
		MatrixBlock mb = Py4jConverterUtils.convertPy4JArrayToMB(buffer.array(), rlen, clen, ValueType.FP32);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertEquals(1.5f, mb.get(0, 0), 0.0001);
		assertEquals(2.5f, mb.get(0, 1), 0.0001);
		assertEquals(3.5f, mb.get(1, 0), 0.0001);
		assertEquals(4.5f, mb.get(1, 1), 0.0001);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testConvertPy4JArrayToMBSparseNotSupported() {
		int rlen = 2;
		int clen = 2;
		byte[] data = {1, 2, 3, 4};
		Py4jConverterUtils.convertPy4JArrayToMB(data, rlen, clen, true, ValueType.UINT8);
	}

	@Test
	public void testConvertSciPyCOOToMB() {
		int rlen = 10;
		int clen = 10;
		int nnz = 3;
		// Create COO format: values at (0,0)=1.0, (1,2)=2.0, (2,1)=3.0
		ByteBuffer dataBuf = ByteBuffer.allocate(Double.BYTES * nnz);
		dataBuf.order(ByteOrder.nativeOrder());
		dataBuf.putDouble(1.0);
		dataBuf.putDouble(2.0);
		dataBuf.putDouble(3.0);
		
		ByteBuffer rowBuf = ByteBuffer.allocate(Integer.BYTES * nnz);
		rowBuf.order(ByteOrder.nativeOrder());
		rowBuf.putInt(0);
		rowBuf.putInt(1);
		rowBuf.putInt(2);
		
		ByteBuffer colBuf = ByteBuffer.allocate(Integer.BYTES * nnz);
		colBuf.order(ByteOrder.nativeOrder());
		colBuf.putInt(0);
		colBuf.putInt(2);
		colBuf.putInt(1);
		
		MatrixBlock mb = Py4jConverterUtils.convertSciPyCOOToMB(
			dataBuf.array(), rowBuf.array(), colBuf.array(), rlen, clen, nnz);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertTrue(mb.isInSparseFormat());
		assertEquals(1.0, mb.get(0, 0), 0.0001);
		assertEquals(2.0, mb.get(1, 2), 0.0001);
		assertEquals(3.0, mb.get(2, 1), 0.0001);
		assertEquals(0.0, mb.get(0, 1), 0.0001);
		assertEquals(0.0, mb.get(1, 0), 0.0001);
	}

	@Test
	public void testConvertSciPyCSRToMB() {
		int rlen = 10;
		int clen = 10;
		int nnz = 3;
		// Create CSR format: values at (0,0)=1.0, (1,2)=2.0, (2,1)=3.0
		ByteBuffer dataBuf = ByteBuffer.allocate(Double.BYTES * nnz);
		dataBuf.order(ByteOrder.nativeOrder());
		dataBuf.putDouble(1.0);
		dataBuf.putDouble(2.0);
		dataBuf.putDouble(3.0);
		
		ByteBuffer indicesBuf = ByteBuffer.allocate(Integer.BYTES * nnz);
		indicesBuf.order(ByteOrder.nativeOrder());
		indicesBuf.putInt(0);  // column for row 0
		indicesBuf.putInt(2);  // column for row 1
		indicesBuf.putInt(1);  // column for row 2
		
		ByteBuffer indptrBuf = ByteBuffer.allocate(Integer.BYTES * (rlen + 1));
		indptrBuf.order(ByteOrder.nativeOrder());
		indptrBuf.putInt(0);  // row 0 starts at index 0
		indptrBuf.putInt(1);  // row 1 starts at index 1
		indptrBuf.putInt(2);  // row 2 starts at index 2
		indptrBuf.putInt(3);  // end marker
		
		MatrixBlock mb = Py4jConverterUtils.convertSciPyCSRToMB(
			dataBuf.array(), indicesBuf.array(), indptrBuf.array(), rlen, clen, nnz);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertTrue(mb.isInSparseFormat());
		assertEquals(1.0, mb.get(0, 0), 0.0001);
		assertEquals(2.0, mb.get(1, 2), 0.0001);
		assertEquals(3.0, mb.get(2, 1), 0.0001);
		assertEquals(0.0, mb.get(0, 1), 0.0001);
		assertEquals(0.0, mb.get(1, 0), 0.0001);
	}

	@Test
	public void testAllocateDenseOrSparseDense() {
		int rlen = 5;
		int clen = 5;
		MatrixBlock mb = Py4jConverterUtils.allocateDenseOrSparse(rlen, clen, false);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertTrue(!mb.isInSparseFormat());
	}

	@Test
	public void testAllocateDenseOrSparseSparse() {
		int rlen = 5;
		int clen = 5;
		MatrixBlock mb = Py4jConverterUtils.allocateDenseOrSparse(rlen, clen, true);
		assertNotNull(mb);
		assertEquals(rlen, mb.getNumRows());
		assertEquals(clen, mb.getNumColumns());
		assertTrue(mb.isInSparseFormat());
	}

	@Test
	public void testAllocateDenseOrSparseLong() {
		long rlen = 10L;
		long clen = 10L;
		MatrixBlock mb = Py4jConverterUtils.allocateDenseOrSparse(rlen, clen, false);
		assertNotNull(mb);
		assertEquals((int) rlen, mb.getNumRows());
		assertEquals((int) clen, mb.getNumColumns());
	}

	@Test(expected = DMLRuntimeException.class)
	public void testAllocateDenseOrSparseLongTooLarge() {
		long rlen = Integer.MAX_VALUE + 1L;
		long clen = 10L;
		Py4jConverterUtils.allocateDenseOrSparse(rlen, clen, false);
	}

	@Test
	public void testConvertMBtoPy4JDenseArr() {
		int rlen = 2;
		int clen = 2;
		MatrixBlock mb = new MatrixBlock(rlen, clen, false);
		mb.allocateBlock();
		mb.set(0, 0, 1.0);
		mb.set(0, 1, 2.0);
		mb.set(1, 0, 3.0);
		mb.set(1, 1, 4.0);
		
		byte[] result = Py4jConverterUtils.convertMBtoPy4JDenseArr(mb);
		assertNotNull(result);
		assertEquals(Double.BYTES * rlen * clen, result.length);
		
		ByteBuffer buffer = ByteBuffer.wrap(result);
		buffer.order(ByteOrder.nativeOrder());
		assertEquals(1.0, buffer.getDouble(), 0.0001);
		assertEquals(2.0, buffer.getDouble(), 0.0001);
		assertEquals(3.0, buffer.getDouble(), 0.0001);
		assertEquals(4.0, buffer.getDouble(), 0.0001);
	}

	@Test
	public void testConvertMBtoPy4JDenseArrRoundTrip() {
		int rlen = 2;
		int clen = 3;
		ByteBuffer buffer = ByteBuffer.allocate(Double.BYTES * rlen * clen);
		buffer.order(ByteOrder.nativeOrder());
		double[] values = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
		for(double val : values) {
			buffer.putDouble(val);
		}
		
		MatrixBlock mb = Py4jConverterUtils.convertPy4JArrayToMB(buffer.array(), rlen, clen);
		byte[] result = Py4jConverterUtils.convertMBtoPy4JDenseArr(mb);
		
		ByteBuffer resultBuffer = ByteBuffer.wrap(result);
		resultBuffer.order(ByteOrder.nativeOrder());
		for(double expected : values) {
			assertEquals(expected, resultBuffer.getDouble(), 0.0001);
		}
	}

	@Test
	public void testConvertMBtoPy4JDenseArrSparseToDense() {
		new Py4jConverterUtils();
		int rlen = 3;
		int clen = 3;
		MatrixBlock mb = new MatrixBlock(rlen, clen, true);
		mb.allocateSparseRowsBlock(false);
		mb.set(0, 0, 1.0);
		mb.set(2, 2, 2.0);
		
		byte[] result = Py4jConverterUtils.convertMBtoPy4JDenseArr(mb);
		assertNotNull(result);
		assertEquals(Double.BYTES * rlen * clen, result.length);
		
		ByteBuffer buffer = ByteBuffer.wrap(result);
		buffer.order(ByteOrder.nativeOrder());
		assertEquals(1.0, buffer.getDouble(), 0.0001);
		// Skip to position (2,2) = index 8
		for(int i = 1; i < 8; i++) {
			assertEquals(0.0, buffer.getDouble(), 0.0001);
		}
		assertEquals(2.0, buffer.getDouble(), 0.0001);
	}
}
