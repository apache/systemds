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
import org.apache.sysds.runtime.io.TensorReaderTextCellParallel;
import org.apache.sysds.runtime.io.TensorWriterTextCellParallel;
import org.apache.sysds.test.TestUtils;

public class TensorTextCellParallelTest {
	static final String FILENAME = "target/testTemp/functions/data/TensorTextCellParallelTest/tensor";
	
	@Test
	public void testReadWriteTextCellParallelBasicTensorFP32() {
		testReadWriteTextCellParallelBasicTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteTextCellParallelBasicTensorFP64() {
		testReadWriteTextCellParallelBasicTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteTextCellParallelBasicTensorINT32() {
		testReadWriteTextCellParallelBasicTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteTextCellParallelBasicTensorINT64() {
		testReadWriteTextCellParallelBasicTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteTextCellParallelBasicTensorBoolean() {
		testReadWriteTextCellParallelBasicTensor(ValueType.BITSET);
	}

	@Test
	public void testReadWriteTextCellParallelBasicTensorString() {
		testReadWriteTextCellParallelBasicTensor(ValueType.STRING);
	}

	private static void testReadWriteTextCellParallelBasicTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createBasicTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadBasicTensorTextCellParallel(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorFP32() {
		testReadWriteTextCellParallelDataTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorFP64() {
		testReadWriteTextCellParallelDataTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorINT32() {
		testReadWriteTextCellParallelDataTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorINT64() {
		testReadWriteTextCellParallelDataTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorBoolean() {
		testReadWriteTextCellParallelDataTensor(ValueType.BITSET);
	}

	@Test
	public void testReadWriteTextCellParallelDataTensorString() {
		testReadWriteTextCellParallelDataTensor(ValueType.STRING);
	}

	private static void testReadWriteTextCellParallelDataTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createDataTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadDataTensorTextCellParallel(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	private static TensorBlock writeAndReadBasicTensorTextCellParallel(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterTextCellParallel writer = new TensorWriterTextCellParallel();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderTextCellParallel reader = new TensorReaderTextCellParallel();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, new ValueType[]{tb1.getValueType()});
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private static TensorBlock writeAndReadDataTensorTextCellParallel(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterTextCellParallel writer = new TensorWriterTextCellParallel();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderTextCellParallel reader = new TensorReaderTextCellParallel();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, tb1.getSchema());
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}
