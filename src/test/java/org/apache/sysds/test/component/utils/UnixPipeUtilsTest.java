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
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

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
					{Types.ValueType.INT64, 4, 64, 55, new MatrixBlock(2, 2, new double[]{0, 1, 2, 3})},
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
			matrixBlock.recomputeNonZeros();

			try (BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.writeNumpyArrayInBatches(out, id, batchSize, numElem, type, matrixBlock);
			}

			double[] output = new double[numElem];
			try (BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				long nonZeros = UnixPipeUtils.readNumpyArrayInBatches(in, id, batchSize, numElem, type, output, 0);
				// Verify nonzero count matches MatrixBlock
				org.junit.Assert.assertEquals(matrixBlock.getNonZeros(), nonZeros);
			}

			assertArrayEquals(matrixBlock.getDenseBlockValues(), output, 1e-9);
		}
	}

	@RunWith(Parameterized.class)
	public static class FrameColumnParameterizedTest {
		@Rule
		public TemporaryFolder folder = new TemporaryFolder();

		@Parameterized.Parameters(name = "{index}: frameType={0}")
		public static Collection<Object[]> data() {
			return Arrays.asList(new Object[][]{
				{Types.ValueType.FP64, new Object[]{1.0, -2.5, 3.25, 4.75}, 64, 201},
				{Types.ValueType.FP32, new Object[]{1.0f, -2.25f, 3.5f, -4.125f}, 48, 202},
				{Types.ValueType.INT32, new Object[]{0, -1, 5, 42}, 32, 203},
				{Types.ValueType.BOOLEAN, new Object[]{true, false, true, false}, 8, 205},
				{Types.ValueType.STRING, new Object[]{"alpha", "beta", "gamma",null, "delta"}, 64, 205},
				{Types.ValueType.STRING, new Object[]{"alphaalphaalphaalphaalpha", "beta", "gamma",null, "delta"}, 16, 205},
				{Types.ValueType.FP64, new Object[]{1.0, -2.5, 3.25, 4.75}, 8, 201},
			});
		}

		private final Types.ValueType type;
		private final Object[] values;
		private final int batchSize;
		private final int id;

		public FrameColumnParameterizedTest(Types.ValueType type, Object[] values, int batchSize, int id) {
			this.type = type;
			this.values = values;
			this.batchSize = batchSize;
			this.id = id;
		}

		@Test
		public void testReadWriteFrameColumn() throws IOException {
			File tempFile = folder.newFile("frame_pipe_" + type.name());
			Array<?> column = createColumn(type, values);

			long bytesWritten;
			try(BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				bytesWritten = UnixPipeUtils.writeFrameColumnToPipe(out, id, batchSize, column, type);
			}

			int totalBytes = Math.toIntExact(bytesWritten);
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				Array<?> read = UnixPipeUtils.readFrameColumnFromPipe(in, id, values.length, totalBytes, batchSize, type);
				assertFrameColumnEquals(column, read, type);
			}
		}

		private static Array<?> createColumn(Types.ValueType type, Object[] values) {
			Array<?> array = ArrayFactory.allocate(type, values.length);
			for(int i = 0; i < values.length; i++) {
				switch(type) {
					case STRING -> array.set(i, (String) values[i]);
					case BOOLEAN -> array.set(i, ((Boolean) values[i]) ? 1.0 : 0.0);
					default -> array.set(i, ((Number) values[i]).doubleValue());
				}
			}
			return array;
		}

		private static void assertFrameColumnEquals(Array<?> expected, Array<?> actual, Types.ValueType type) {
			org.junit.Assert.assertEquals(expected.size(), actual.size());
			for(int i = 0; i < expected.size(); i++) {
				switch(type) {
					case FP64 -> org.junit.Assert.assertEquals(
						((Number) expected.get(i)).doubleValue(),
						((Number) actual.get(i)).doubleValue(), 1e-9);
					case FP32 -> org.junit.Assert.assertEquals(
						((Number) expected.get(i)).floatValue(),
						((Number) actual.get(i)).floatValue(), 1e-6f);
					case STRING -> org.junit.Assert.assertEquals(expected.get(i), actual.get(i));
					default -> org.junit.Assert.assertEquals(expected.get(i), actual.get(i));
				}
			}
		}
	}

	public static class NonParameterizedTest {
		@Rule
		public TemporaryFolder folder = new TemporaryFolder();

		@Test(expected = FileNotFoundException.class)
		public void testOpenInputFileNotFound() throws IOException {
			new UnixPipeUtils();
			File nonExistentFile = new File(folder.getRoot(), "nonexistent.pipe");
			UnixPipeUtils.openInput(nonExistentFile.getAbsolutePath(), 123);
		}

		@Test(expected = FileNotFoundException.class)
		public void testOpenOutputFileNotFound() throws IOException {
			File nonExistentFile = new File(folder.getRoot(), "nonexistent.pipe");
			UnixPipeUtils.openOutput(nonExistentFile.getAbsolutePath(), 123);
		}


		@Test
		public void testOpenInputAndOutputHandshakeMatch() throws IOException {
			File tempFile = folder.newFile("pipe_test1");
			int id = 42;
			
			try (BufferedOutputStream bos = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {}
			try (BufferedInputStream bis = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {}
		}

		@Test(expected = IllegalStateException.class)
		public void testOpenInputHandshakeMismatch() throws IOException {
			File tempFile = folder.newFile("pipe_test2");
			int writeId = 123;
			int wrongReadId = 456;

			try (BufferedOutputStream bos = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), writeId)) {}
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

		@Test(expected = IOException.class)
		public void testReadColumnFromPipeError() throws IOException {
			File tempFile = folder.newFile("pipe_test3");
			int id = 42;
			
			BufferedOutputStream bos = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id);
			BufferedInputStream bis = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id);
			Array<?> column = ArrayFactory.allocate(Types.ValueType.INT64, 4);
			UnixPipeUtils.writeFrameColumnToPipe(bos, id, 16, column, Types.ValueType.INT64);
			UnixPipeUtils.readFrameColumnFromPipe(bis, 42, 4, 12, 32 * 1024, Types.ValueType.INT32);
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

		@Test(expected = UnsupportedOperationException.class)
		public void testGetElementSizeUnsupportedType() {
			UnixPipeUtils.getElementSize(Types.ValueType.STRING);
		}

		@Test(expected = UnsupportedOperationException.class)
		public void testReadNumpyArrayUnsupportedType() throws IOException {
			File file = folder.newFile("unsupported_type.pipe");
			int id = 7;
			try (BufferedOutputStream out = UnixPipeUtils.openOutput(file.getAbsolutePath(), id)) {
				UnixPipeUtils.writeHandshake(id, out); // start handshake
				out.flush();
				UnixPipeUtils.writeHandshake(id, out); // end handshake, no payload
			}
			double[] outArr = new double[0];
			try (BufferedInputStream in = UnixPipeUtils.openInput(file.getAbsolutePath(), id)) {
				UnixPipeUtils.readNumpyArrayInBatches(in, id, 32, 0, Types.ValueType.STRING, outArr, 0);
			}
		}

		@Test(expected = NullPointerException.class)
		public void testWriteNumpyArrayInBatchesError() throws IOException {
			UnixPipeUtils.writeNumpyArrayInBatches(null, 0, 0, 0, Types.ValueType.INT32, null);
		}

		@Test
		public void testGetBufferReaderUnsupportedType() throws Exception {
			Method m = UnixPipeUtils.class.getDeclaredMethod("getBufferReader", Types.ValueType.class);
			m.setAccessible(true);

			try {
				m.invoke(null, Types.ValueType.STRING);
				org.junit.Assert.fail("Expected UnsupportedOperationException");
			} catch (InvocationTargetException e) {
				org.junit.Assert.assertTrue(e.getCause() instanceof UnsupportedOperationException);
			}
		}

		@Test
		public void testGetBufferWriterUnsupportedType() throws Exception {
			Method m = UnixPipeUtils.class.getDeclaredMethod("getBufferWriter", Types.ValueType.class);
			m.setAccessible(true);

			try {
				m.invoke(null, Types.ValueType.STRING);
				org.junit.Assert.fail("Expected UnsupportedOperationException");
			} catch (InvocationTargetException e) {
				org.junit.Assert.assertTrue(e.getCause() instanceof UnsupportedOperationException);
			}
		}

		@Test(expected = UnsupportedOperationException.class)
		public void testReadWriteFrameColumnUINT8() throws IOException {
			File tempFile = folder.newFile("frame_pipe_UINT8");
			int id  = 204;

			BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id);
			UnixPipeUtils.writeHandshake(id, out);
			out.write(new byte[]{(byte) 0x00, (byte) 0x01, (byte) 0x02, (byte) 0x03});
			UnixPipeUtils.writeHandshake(id, out);
			out.flush();

			Array<?> read = null;
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				read = UnixPipeUtils.readFrameColumnFromPipe(in, id, 4, 4, 32 * 1024, Types.ValueType.UINT8);
				for(int i = 0; i < 4; i++) {
					org.junit.Assert.assertEquals(i, read.get(i));
				}
			}
			try(BufferedOutputStream out2 = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.writeFrameColumnToPipe(out2, id, 16, read, Types.ValueType.UINT8);
			}
		}

		@Test
		public void testReadWriteFrameColumnINT64() throws IOException {
			File tempFile = folder.newFile("frame_pipe_INT32");
			int id  = 204;

			BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id);
			UnixPipeUtils.writeHandshake(id, out);
			// write 4 int64 values
			ByteBuffer bb = ByteBuffer.allocate(8 * 4).order(ByteOrder.LITTLE_ENDIAN);
			for(int i = 0; i < 4; i++) {
				bb.putLong(i);
			}
			out.write(bb.array());
			UnixPipeUtils.writeHandshake(id, out);
			out.flush();

			Array<?> read = null;
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				read = UnixPipeUtils.readFrameColumnFromPipe(in, id, 4, 32, 32 * 1024, Types.ValueType.INT64);
				for(int i = 0; i < 4; i++) {
					org.junit.Assert.assertEquals(i, ((Number) read.get(i)).longValue());
				}
			}
		}

		@Test(expected = UnsupportedOperationException.class)
		public void testReadWriteFrameColumnUnsupportedType() throws IOException {
			File tempFile = folder.newFile("frame_pipe_HASH64");
			int id  = 204;

			BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id);
			UnixPipeUtils.writeHandshake(id, out);
			out.flush();

			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.readFrameColumnFromPipe(in, id, 4, 32, 4, Types.ValueType.HASH64);
			}
		}

		@Test
		public void testReadWriteFrameColumnLongString1() throws IOException {
			File tempFile = folder.newFile("frame_pipe_long_string");
			Array<?> column = FrameColumnParameterizedTest.createColumn(Types.ValueType.STRING, 
				new Object[]{"alphaalphaalphaalphaalphaa", "beta", "gamma",null, "delta"});
			int id = 205;
			int batchSize = 16;

			long bytesWritten;
			try(BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				bytesWritten = UnixPipeUtils.writeFrameColumnToPipe(out, id, batchSize, column, Types.ValueType.STRING);
			}

			int totalBytes = Math.toIntExact(bytesWritten);
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				Array<?> read = UnixPipeUtils.readFrameColumnFromPipe(in, id, column.size(), totalBytes, batchSize, Types.ValueType.STRING);
				FrameColumnParameterizedTest.assertFrameColumnEquals(column, read, Types.ValueType.STRING);
			}
		}

		@Test
		public void testReadWriteFrameColumnLongString2() throws IOException {
			File tempFile = folder.newFile("frame_pipe_long_string");
			StringBuilder sb = new StringBuilder();
			for(int i = 0; i < 35*1024; i++) {
				sb.append("a");
			}
			Array<?> column = FrameColumnParameterizedTest.createColumn(Types.ValueType.STRING, 
				new Object[]{sb.toString()});
			int id = 205;
			int batchSize = 16*1024;

			long bytesWritten;
			try(BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				bytesWritten = UnixPipeUtils.writeFrameColumnToPipe(out, id, batchSize, column, Types.ValueType.STRING);
			}

			int totalBytes = Math.toIntExact(bytesWritten);
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				Array<?> read = UnixPipeUtils.readFrameColumnFromPipe(in, id, column.size(), totalBytes, batchSize, Types.ValueType.STRING);
				FrameColumnParameterizedTest.assertFrameColumnEquals(column, read, Types.ValueType.STRING);
			}
		}

		@Test
		public void testReadWriteFrameColumnString() throws IOException {
			File tempFile = folder.newFile("frame_pipe_long_string");
			Array<?> column = FrameColumnParameterizedTest.createColumn(Types.ValueType.STRING, 
				new Object[]{"alphabet"});
			int id = 205;
			int batchSize = 12;

			long bytesWritten;
			try(BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				bytesWritten = UnixPipeUtils.writeFrameColumnToPipe(out, id, batchSize, column, Types.ValueType.STRING);
			}

			int totalBytes = Math.toIntExact(bytesWritten);
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				Array<?> read = UnixPipeUtils.readFrameColumnFromPipe(in, id, column.size(), totalBytes, batchSize, Types.ValueType.STRING);
				FrameColumnParameterizedTest.assertFrameColumnEquals(column, read, Types.ValueType.STRING);
			}
		}

		@Test
		public void testWriteFrameColumnINT32() throws IOException {
			File tempFile = folder.newFile("frame_pip2_INT32");
			int id  = 204;
			Array<?> column = FrameColumnParameterizedTest.createColumn(Types.ValueType.INT32, 
				new Object[]{0, 1, 2, 3});

			try(BufferedOutputStream out = UnixPipeUtils.openOutput(tempFile.getAbsolutePath(), id)) {
				UnixPipeUtils.writeFrameColumnToPipe(out, id, 4, column, Types.ValueType.INT32);
			}

			Array<?> read = null;
			try(BufferedInputStream in = UnixPipeUtils.openInput(tempFile.getAbsolutePath(), id)) {
				read = UnixPipeUtils.readFrameColumnFromPipe(in, id, 4, 16, 4, Types.ValueType.INT32);
				FrameColumnParameterizedTest.assertFrameColumnEquals(column, read, Types.ValueType.INT32);
			}

		}
	}
}
