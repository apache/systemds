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
import org.apache.sysds.runtime.io.TensorReaderBinaryBlockParallel;
import org.apache.sysds.runtime.io.TensorWriterBinaryBlockParallel;
import org.apache.sysds.test.TestUtils;

public class TensorBinaryBlockParallelTest {
	static final String FILENAME = "target/testTemp/functions/data/TensorBinaryBlockParallelTest/tensor";
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorFP32() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.FP32);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorFP64() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.FP64);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorINT32() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.INT32);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorINT64() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.INT64);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorBoolean() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.BOOLEAN);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelBasicTensorString() {
		testReadWriteBinaryBlockParallelBasicTensor(ValueType.STRING);
	}
	
	private static void testReadWriteBinaryBlockParallelBasicTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createBasicTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadBasicTensorBinaryBlockParallel(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorFP32() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.FP32);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorFP64() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.FP64);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorINT32() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.INT32);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorINT64() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.INT64);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorBoolean() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.BOOLEAN);
	}
	
	@Test
	public void testReadWriteBinaryBlockParallelDataTensorString() {
		testReadWriteBinaryBlockParallelDataTensor(ValueType.STRING);
	}
	
	private static void testReadWriteBinaryBlockParallelDataTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createDataTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadDataTensorBinaryBlockParallel(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}
	
	private static TensorBlock writeAndReadBasicTensorBinaryBlockParallel(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterBinaryBlockParallel writer = new TensorWriterBinaryBlockParallel();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderBinaryBlockParallel reader = new TensorReaderBinaryBlockParallel();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, new ValueType[]{tb1.getValueType()});
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	private static TensorBlock writeAndReadDataTensorBinaryBlockParallel(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterBinaryBlockParallel writer = new TensorWriterBinaryBlockParallel();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderBinaryBlockParallel reader = new TensorReaderBinaryBlockParallel();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, tb1.getSchema());
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}
