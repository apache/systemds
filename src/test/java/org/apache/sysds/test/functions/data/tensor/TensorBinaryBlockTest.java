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

package org.apache.sysds.test.functions.data.tensor;

import org.junit.Test;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.io.TensorReaderBinaryBlock;
import org.apache.sysds.runtime.io.TensorWriterBinaryBlock;
import org.apache.sysds.test.TestUtils;

public class TensorBinaryBlockTest {
	static final String FILENAME = "target/testTemp/functions/data/TensorBinaryBlockTest/tensor";
	
	@Test
	public void testReadWriteBinaryBlockBasicTensorFP32() {
		testReadWriteBinaryBlockBasicTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteBinaryBlockBasicTensorFP64() {
		testReadWriteBinaryBlockBasicTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteBinaryBlockBasicTensorINT32() {
		testReadWriteBinaryBlockBasicTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteBinaryBlockBasicTensorINT64() {
		testReadWriteBinaryBlockBasicTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteBinaryBlockBasicTensorBoolean() {
		testReadWriteBinaryBlockBasicTensor(ValueType.BOOLEAN);
	}

	@Test
	public void testReadWriteBinaryBlockBasicTensorString() {
		testReadWriteBinaryBlockBasicTensor(ValueType.STRING);
	}

	private static void testReadWriteBinaryBlockBasicTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createBasicTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadBasicTensorBinaryBlock(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorFP32() {
		testReadWriteBinaryBlockDataTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorFP64() {
		testReadWriteBinaryBlockDataTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorINT32() {
		testReadWriteBinaryBlockDataTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorINT64() {
		testReadWriteBinaryBlockDataTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorBoolean() {
		testReadWriteBinaryBlockDataTensor(ValueType.BOOLEAN);
	}

	@Test
	public void testReadWriteBinaryBlockDataTensorString() {
		testReadWriteBinaryBlockDataTensor(ValueType.STRING);
	}

	private static void testReadWriteBinaryBlockDataTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createDataTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadDataTensorBinaryBlock(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	private static TensorBlock writeAndReadBasicTensorBinaryBlock(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterBinaryBlock writer = new TensorWriterBinaryBlock();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderBinaryBlock reader = new TensorReaderBinaryBlock();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, new ValueType[]{tb1.getValueType()});
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private static TensorBlock writeAndReadDataTensorBinaryBlock(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterBinaryBlock writer = new TensorWriterBinaryBlock();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderBinaryBlock reader = new TensorReaderBinaryBlock();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, tb1.getSchema());
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}
