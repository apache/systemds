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
import org.apache.sysds.runtime.io.TensorReaderTextCell;
import org.apache.sysds.runtime.io.TensorWriterTextCell;
import org.apache.sysds.test.TestUtils;

public class TensorTextCellTest {
	static final String FILENAME = "target/testTemp/functions/data/TensorTextCellTest/tensor";
	
	@Test
	public void testReadWriteTextCellBasicTensorFP32() {
		testReadWriteTextCellBasicTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteTextCellBasicTensorFP64() {
		testReadWriteTextCellBasicTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteTextCellBasicTensorINT32() {
		testReadWriteTextCellBasicTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteTextCellBasicTensorINT64() {
		testReadWriteTextCellBasicTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteTextCellBasicTensorBoolean() {
		testReadWriteTextCellBasicTensor(ValueType.BOOLEAN);
	}

	@Test
	public void testReadWriteTextCellBasicTensorString() {
		TensorBlock tb1 = TestUtils.createBasicTensor(ValueType.STRING, 70, 30, 0.7);
		tb1.set(new int[]{0, 0}, "\"  f  f  \"");
		tb1.set(new int[]{69, 29}, "respect");
		TensorBlock tb2 = writeAndReadBasicTensorTextCell(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	private static void testReadWriteTextCellBasicTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createBasicTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadBasicTensorTextCell(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	@Test
	public void testReadWriteTextCellDataTensorFP32() {
		testReadWriteTextCellDataTensor(ValueType.FP32);
	}

	@Test
	public void testReadWriteTextCellDataTensorFP64() {
		testReadWriteTextCellDataTensor(ValueType.FP64);
	}

	@Test
	public void testReadWriteTextCellDataTensorINT32() {
		testReadWriteTextCellDataTensor(ValueType.INT32);
	}

	@Test
	public void testReadWriteTextCellDataTensorINT64() {
		testReadWriteTextCellDataTensor(ValueType.INT64);
	}

	@Test
	public void testReadWriteTextCellDataTensorBoolean() {
		testReadWriteTextCellDataTensor(ValueType.BOOLEAN);
	}

	@Test
	public void testReadWriteTextCellDataTensorString() {
		TensorBlock tb1 = TestUtils.createDataTensor(ValueType.STRING, 70, 30, 0.7);
		tb1.set(new int[]{0, 0}, "\"  f  f  \"");
		tb1.set(new int[]{69, 29}, "respect");
		TensorBlock tb2 = writeAndReadDataTensorTextCell(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	private static void testReadWriteTextCellDataTensor(ValueType vt) {
		TensorBlock tb1 = TestUtils.createDataTensor(vt, 70, 3000, 0.7);
		TensorBlock tb2 = writeAndReadDataTensorTextCell(tb1);
		TestUtils.compareTensorBlocks(tb1, tb2);
	}

	private static TensorBlock writeAndReadBasicTensorTextCell(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterTextCell writer = new TensorWriterTextCell();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderTextCell reader = new TensorReaderTextCell();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, new ValueType[]{tb1.getValueType()});
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private static TensorBlock writeAndReadDataTensorTextCell(TensorBlock tb1) {
		try {
			long[] dims = tb1.getLongDims();
			TensorWriterTextCell writer = new TensorWriterTextCell();
			writer.writeTensorToHDFS(tb1, FILENAME, 1024);
			TensorReaderTextCell reader = new TensorReaderTextCell();
			return reader.readTensorFromHDFS(FILENAME, dims, 1024, tb1.getSchema());
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}
