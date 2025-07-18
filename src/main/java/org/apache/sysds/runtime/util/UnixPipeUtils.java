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

package org.apache.sysds.runtime.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

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
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class UnixPipeUtils {
	private static final Log LOG = LogFactory.getLog(UnixPipeUtils.class.getName());

	/**
	 * Opens a named pipe for input, reads 4 bytes as an int, compares it to the expected ID.
	 * If matched, returns the InputStream for further use.
	 *
	 * @param pipePath The filesystem path to the FIFO pipe
	 * @param expectedId The expected handshake ID
	 * @return BufferedInputStream if handshake succeeds
	 * @throws IOException if file access fails
	 * @throws IllegalStateException if handshake ID doesn't match
	 */

	public static BufferedInputStream openInput(String pipePath, int expectedId) throws IOException {
		File pipeFile = new File(pipePath);
		if (!pipeFile.exists()) {
			throw new FileNotFoundException("Pipe not found at path: " + pipePath);
		}

		FileInputStream fis = new FileInputStream(pipeFile);
		BufferedInputStream bis = new BufferedInputStream(fis);

		readHandshake(expectedId, bis);

		return bis;
	}

	public static void readHandshake(int expectedId, BufferedInputStream bis) throws IOException {
		// Read 4 bytes for handshake
		byte[] buffer = new byte[4];
		int bytesRead = bis.read(buffer);
		if (bytesRead != 4) {
			bis.close();
			throw new IOException("Failed to read handshake integer from pipe");
		}

		// Convert bytes to int (assuming little-endian to match typical Python struct.pack)
		int receivedId = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getInt();
		expectedId += 1000;

		if (receivedId != expectedId) {
			bis.close();
			throw new IllegalStateException("Handshake ID mismatch: expected " + expectedId + ", got " + receivedId);
		}
	}

	public static BufferedOutputStream openOutput(String pipePath, int expectedId) throws IOException {
		File pipeFile = new File(pipePath);
		if (!pipeFile.exists()) {
			throw new FileNotFoundException("Pipe not found at path: " + pipePath);
		}

		FileOutputStream fos = new FileOutputStream(pipeFile);
		BufferedOutputStream bos = new BufferedOutputStream(fos);

		writeHandshake(expectedId, bos);

		return bos;
	}

	public static void writeHandshake(int expectedId, BufferedOutputStream bos) throws IOException {
		// Convert int to 4-byte little-endian and send as handshake
		byte[] handshake = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(expectedId + 1000).array();
		bos.write(handshake);
		bos.flush();
	}

	public static void readNumpyArrayInBatches(BufferedInputStream in, int id, int batchSize, int numElem,
											   Types.ValueType type, double[] out, int offsetOut)
			throws IOException {
		int elemSize;
		switch (type){
			case UINT8 -> elemSize = 1;
			case INT32, FP32 -> elemSize = 4;
			default -> elemSize = 8;
		}

		try {
			// Read start header
			readHandshake(id, in);
			long bytesRemaining = ((long) numElem) * elemSize;
			byte[] buffer = new byte[batchSize];

			while (bytesRemaining > 0) {
				int currentBatchSize = (int) Math.min(batchSize, bytesRemaining);
				int totalRead = 0;

				while (totalRead < currentBatchSize) {
					int bytesRead = in.read(buffer, totalRead, currentBatchSize - totalRead);
					if (bytesRead == -1) {
						throw new EOFException("Unexpected end of stream in pipe #" + id +
								": expected " + currentBatchSize + " bytes, got " + totalRead);
					}
					totalRead += bytesRead;
				}

				// Interpret bytes with value type and fill the dense MB
				offsetOut = fillDoubleArrayFromByteArray(type, out, offsetOut, buffer, currentBatchSize);
				bytesRemaining -= currentBatchSize;
			}

			// Read end header
			readHandshake(id, in);

		} catch (Exception e) {
			LOG.error("Error occurred while reading data from pipe #" + id, e);
			throw e;
		}
	}

	private static int fillDoubleArrayFromByteArray(Types.ValueType type, double[] out, int offsetOut, byte[] buffer,
													int currentBatchSize) {
		ByteBuffer bb = ByteBuffer.wrap(buffer, 0, currentBatchSize).order(ByteOrder.LITTLE_ENDIAN);
		switch (type){
			default -> {
				DoubleBuffer doubleBuffer = bb.asDoubleBuffer();
				int numDoubles = doubleBuffer.remaining();
				doubleBuffer.get(out, offsetOut, numDoubles);
				offsetOut += numDoubles;
			}
			case FP32 -> {
				FloatBuffer floatBuffer = bb.asFloatBuffer();
				int numFloats = floatBuffer.remaining();
				for (int i = 0; i < numFloats; i++) {
					out[offsetOut++] = floatBuffer.get();
				}
			}
			case INT32 -> {
				IntBuffer intBuffer = bb.asIntBuffer();
				int numInts = intBuffer.remaining();
				for (int i = 0; i < numInts; i++) {
					out[offsetOut++] = intBuffer.get();
				}
			}
			case UINT8 -> {
				for (int i = 0; i < currentBatchSize; i++) {
					out[offsetOut++] = bb.get(i) & 0xFF;
				}
			}
		}
		return offsetOut;
	}

	public static long writeNumpyArrayInBatches(BufferedOutputStream out, int id, int batchSize, int numElem,
												Types.ValueType type, MatrixBlock mb) throws IOException {
		int elemSize;
		switch (type) {
			case UINT8 -> elemSize = 1;
			case INT32, FP32 -> elemSize = 4;
			default -> elemSize = 8;
		}
		long totalBytesWritten = 0;

		// Write start header
		writeHandshake(id, out);

		int bytesRemaining = numElem * elemSize;
		int offset = 0;

		byte[] buffer = new byte[batchSize];

		while (bytesRemaining > 0) {
			int currentBatchSize = Math.min(batchSize, bytesRemaining);

			// Fill buffer from MatrixBlock into byte[] (typed)
			int bytesWritten = fillByteArrayFromDoubleArray(type, mb, offset, buffer, currentBatchSize);
			totalBytesWritten += bytesWritten;
//			if (bytesWritten != currentBatchSize) {
//				throw new IOException("Internal error: mismatched buffer fill size");
//			}

			out.write(buffer, 0, currentBatchSize);
			offset += currentBatchSize / elemSize;
			bytesRemaining -= currentBatchSize;
		}

		out.flush();

		// Write end header
		writeHandshake(id, out);
		return totalBytesWritten;
	}

	private static int fillByteArrayFromDoubleArray(Types.ValueType type, MatrixBlock mb, int offsetIn,
													byte[] buffer, int maxBytes) {
		ByteBuffer bb = ByteBuffer.wrap(buffer, 0, maxBytes).order(ByteOrder.LITTLE_ENDIAN);
		int r,c;
		switch (type) {
			default -> { // FP64
				DoubleBuffer doubleBuffer = bb.asDoubleBuffer();
				int count = Math.min(doubleBuffer.remaining(), mb.getNumRows() * mb.getNumColumns() - offsetIn);
				for (int i = 0; i < count; i++) {
					r = (offsetIn + i) / mb.getNumColumns();
					c = (offsetIn + i) % mb.getNumColumns();
					doubleBuffer.put(mb.getDouble(r,c));
				}
				return count * 8;
			}
			case FP32 -> {
				FloatBuffer floatBuffer = bb.asFloatBuffer();
				int count = Math.min(floatBuffer.remaining(), mb.getNumRows() * mb.getNumColumns() - offsetIn);
				for (int i = 0; i < count; i++) {
					r = (offsetIn + i) / mb.getNumColumns();
					c = (offsetIn + i) % mb.getNumColumns();
					floatBuffer.put((float) mb.getDouble(r,c));
				}
				return count * 4;
			}
			case INT32 -> {
				IntBuffer intBuffer = bb.asIntBuffer();
				int count = Math.min(intBuffer.remaining(), mb.getNumRows() * mb.getNumColumns() - offsetIn);
				for (int i = 0; i < count; i++) {
					r = (offsetIn + i) / mb.getNumColumns();
					c = (offsetIn + i) % mb.getNumColumns();
					intBuffer.put((int) mb.getDouble(r,c));
				}
				return count * 4;
			}
			case UINT8 -> {
				int count = Math.min(maxBytes, mb.getNumRows() * mb.getNumColumns() - offsetIn);
				for (int i = 0; i < count; i++) {
					r = (offsetIn + i) / mb.getNumColumns();
					c = (offsetIn + i) % mb.getNumColumns();
					buffer[i] = (byte) ((int) mb.getDouble(r,c) & 0xFF);
				}
				return count;
			}
		}
	}
}