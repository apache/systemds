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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UnixPipeUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertArrayEquals;


@RunWith(Enclosed.class)
public class UnixPipeUtilsTest {

	@RunWith(Parameterized.class)
	public static class ParameterizedTest {
		@Rule
		public TemporaryFolder folder = new TemporaryFolder();

		@Parameterized.Parameters(name = "{index}: type={0}")
		public static Collection<Object[]> data() {
			return Arrays.asList(new Object[][]{
					{Types.ValueType.FP64, 6, 48, 99, new MatrixBlock(2, 3, new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})},
					{Types.ValueType.FP32, 6, 24, 88, new MatrixBlock(3, 2, new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})},
					{Types.ValueType.INT32, 4, 16, 77, new MatrixBlock(2, 2, new double[]{0, -1, 2, -3})},
					{Types.ValueType.UINT8, 4, 4, 66, new MatrixBlock(2, 2, new double[]{0, 1, 2, 3})}
			});
		}

		private final Types.ValueType type;
		private final int numElem;
		private final int batchSize;
		private final int id;
		private final MatrixBlock matrixBlock;


		public ParameterizedTest(Types.ValueType type, int numElem, int batchSize, int id, MatrixBlock matrixBlock) {
			this.type = type;
			this.numElem = numElem;
			this.batchSize = batchSize;
			this.id = id;
			this.matrixBlock = matrixBlock;
		}

		@Test
		public void testReadWriteNumpyArrayBatch() throws IOException {
			File tempFile = folder.newFile("pipe_test_" + type.name());

			try (BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.writeNumpyArrayInBatches(out, id, batchSize, numElem, type, matrixBlock);
			}

			double[] output = new double[numElem];
			try (BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.readNumpyArrayInBatches(in, id, batchSize, numElem, type, output, 0);
			}

			assertArrayEquals(matrixBlock.getDenseBlockValues(), output, 1e-9);
		}
	}

	public static class NonParameterizedTest {
		@Rule
		public TemporaryFolder folder = new TemporaryFolder();

		@Test(expected = FileNotFoundException.class)
		public void testOpenInputFileNotFound() throws IOException {
			// instantiate class once for coverage
			new UnixPipeUtils();

			// Create a path that does not exist
			File nonExistentFile = new File(folder.getRoot(), "nonexistent.pipe");

			// This should throw FileNotFoundException
			UnixPipeUtils.openInput(nonExistentFile.getAbsolutePath(), 123);
		}

		@Test(expected = FileNotFoundException.class)
		public void testOpenOutputFileNotFound() throws IOException {
			// Create a path that does not exist
			File nonExistentFile = new File(folder.getRoot(), "nonexistent.pipe");

			// This should throw FileNotFoundException
			UnixPipeUtils.openOutput(nonExistentFile.getAbsolutePath(), 123);
		}


		@Test
		public void testOpenInputAndOutputHandshakeMatch() throws IOException {
			File tempFile = folder.newFile("pipe_test1");
			int id = 42;

			// Write expected handshake
			try (BufferedOutputStream bos = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {}

			// Read and validate handshake
			try (BufferedInputStream bis = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				// success: no exception = handshake passed
			}
		}

		@Test(expected = IllegalStateException.class)
		public void testOpenInputHandshakeMismatch() throws IOException {
			File tempFile = folder.newFile("pipe_test2");
			int writeId = 123;
			int wrongReadId = 456;

			try (BufferedOutputStream bos = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), writeId)) {}

			// Will throw due to ID mismatch
			UnixPipeUtils.openInput(tempFile.getAbsolutePath(), wrongReadId);
		}

		@Test(expected = IOException.class)
		public void testOpenInputIncompleteHandshake() throws IOException {
			File tempFile = folder.newFile("short_handshake.pipe");

			// Write only 2 bytes instead of 4
			try (FileOutputStream fos = new FileOutputStream(tempFile)) {
				fos.write(new byte[]{0x01, 0x02});
			}

			UnixPipeUtils.openInput(tempFile.getAbsolutePath(), 100);
		}

		@Test(expected = EOFException.class)
		public void testReadNumpyArrayUnexpectedEOF() throws IOException {
			File tempFile = folder.newFile("pipe_test5");
			int id = 12;
			int numElem = 5;
			int batchSize = 40;
			Types.ValueType type = Types.ValueType.FP64;

			// Write partial data (handshake + 3 doubles instead of 5)
			try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(tempFile))) {
				ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(id + 1000);
				out.write(bb.array());

				// Write 3 doubles only
				bb = ByteBuffer.allocate(8 * 3).order(ByteOrder.LITTLE_ENDIAN);
				for (int i = 0; i < 3; i++)
					bb.putDouble(i + 1.0);
				out.write(bb.array());

				// no end handshake
				out.flush();
			}

			double[] outArr = new double[numElem];
			try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(tempFile))) {
				UnixPipeUtils.readNumpyArrayInBatches(in, id, batchSize, numElem, type, outArr, 0);
			}
		}
	}
}
