/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.	 See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.util;

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
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;


public class UnixPipeUtils {
	private static final Log LOG = LogFactory.getLog(UnixPipeUtils.class.getName());

	public static int getElementSize(Types.ValueType type) {
		return switch (type) {
			case UINT8, BOOLEAN -> 1;
			case INT32, FP32 -> 4;
			case INT64, FP64 -> 8;
			default -> throw new UnsupportedOperationException("Unsupported type: " + type);
		};
	}

	private static ByteBuffer newLittleEndianBuffer(byte[] buffer, int length) {
		return ByteBuffer.wrap(buffer, 0, length).order(ByteOrder.LITTLE_ENDIAN);
	}

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
		compareHandshakeIds(expectedId, bis, buffer);
	}

	private static void compareHandshakeIds(int expectedId, BufferedInputStream bis, byte[] buffer) throws IOException {
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

	@FunctionalInterface
	private interface BufferReader {
		int readTo(Object dest, int offset, ByteBuffer bb);
	}

	private static BufferReader getBufferReader(Types.ValueType type) {
		return switch (type) {
			case FP64 -> (dest, offset, bb) -> {
				DoubleBuffer db = bb.asDoubleBuffer();
				double[] out = (double[]) dest;
				int remaining = db.remaining();
				db.get(out, offset, remaining);
				return offset + remaining;
			};
			case FP32 -> (dest, offset, bb) -> {
				FloatBuffer fb = bb.asFloatBuffer();
				double[] out = (double[]) dest;
				int n = fb.remaining();
				for (int i = 0; i < n; i++) out[offset++] = fb.get();
				return offset;
			};
			case INT64 -> (dest, offset, bb) -> {
				LongBuffer lb = bb.asLongBuffer();
				double[] out = (double[]) dest;
				int n = lb.remaining();
				for (int i = 0; i < n; i++) out[offset++] = lb.get();
				return offset;
			};
			case INT32 -> (dest, offset, bb) -> {
				IntBuffer ib = bb.asIntBuffer();
				double[] out = (double[]) dest;
				int n = ib.remaining();
				for (int i = 0; i < n; i++) out[offset++] = ib.get();
				return offset;
			};
			case UINT8 -> (dest, offset, bb) -> {
				double[] out = (double[]) dest;
				for (int i = 0; i < bb.limit(); i++) out[offset++] = bb.get(i) & 0xFF;
				return offset;
			};
			default -> throw new UnsupportedOperationException("Unsupported type: " + type);
		};
	}

	private static void readFully(BufferedInputStream in, byte[] buffer, int len) throws IOException {
		int total = 0;
		while (total < len) {
			int read = in.read(buffer, total, len - total);
			if (read == -1)
				throw new EOFException("Unexpected end of stream");
			total += read;
		}
	}

	public static long readNumpyArrayInBatches(BufferedInputStream in, int id, int batchSize, int numElem,
		Types.ValueType type, double[] out, int offsetOut)
			throws IOException {
		int elemSize = getElementSize(type);
		long nonZeros = 0;

		try {
			// Read start header
			readHandshake(id, in);
			long bytesRemaining = ((long) numElem) * elemSize;
			byte[] buffer = new byte[batchSize];

			BufferReader reader = getBufferReader(type);
			int prevOffset = offsetOut;
			while (bytesRemaining > 0) {
				int chunk = (int) Math.min(batchSize, bytesRemaining);
				readFully(in, buffer, chunk);
				offsetOut = reader.readTo(out, offsetOut, newLittleEndianBuffer(buffer, chunk));
				
				// Count nonzeros in the batch we just read (performant: single pass)
				for (int i = prevOffset; i < offsetOut; i++) {
					if (out[i] != 0.0) {
						nonZeros++;
					}
				}
				prevOffset = offsetOut;
				bytesRemaining -= chunk;
			}

			// Read end header
			readHandshake(id, in);
			return nonZeros;

		} catch (Exception e) {
			LOG.error("Error occurred while reading data from pipe #" + id, e);
			throw e;
		}
	}


	@FunctionalInterface
	private interface BufferWriter {
		int writeFrom(Object src, int offset, ByteBuffer bb);
	}

	private static BufferWriter getBufferWriter(Types.ValueType type) {
		return switch (type) {
			case FP64 -> (src, offset, bb) -> {
				MatrixBlock mb = (MatrixBlock) src;
				DoubleBuffer db = bb.asDoubleBuffer();
				int n = Math.min(db.remaining(), mb.getNumRows() * mb.getNumColumns() - offset);
				for (int i = 0; i < n; i++) {
					int r = (offset + i) / mb.getNumColumns();
					int c = (offset + i) % mb.getNumColumns();
					db.put(mb.getDouble(r, c));
				}
				return n * 8;
			};
			case FP32 -> (src, offset, bb) -> {
				MatrixBlock mb = (MatrixBlock) src;
				FloatBuffer fb = bb.asFloatBuffer();
				int n = Math.min(fb.remaining(), mb.getNumRows() * mb.getNumColumns() - offset);
				for (int i = 0; i < n; i++) {
					int r = (offset + i) / mb.getNumColumns();
					int c = (offset + i) % mb.getNumColumns();
					fb.put((float) mb.getDouble(r, c));
				}
				return n * 4;
			};
			case INT64 -> (src, offset, bb) -> {
				MatrixBlock mb = (MatrixBlock) src;
				LongBuffer lb = bb.asLongBuffer();
				int n = Math.min(lb.remaining(), mb.getNumRows() * mb.getNumColumns() - offset);
				for (int i = 0; i < n; i++) {
					int r = (offset + i) / mb.getNumColumns();
					int c = (offset + i) % mb.getNumColumns();
					lb.put((long) mb.getDouble(r, c));
				}
				return n * 8;
			};
			case INT32 -> (src, offset, bb) -> {
				MatrixBlock mb = (MatrixBlock) src;
				IntBuffer ib = bb.asIntBuffer();
				int n = Math.min(ib.remaining(), mb.getNumRows() * mb.getNumColumns() - offset);
				for (int i = 0; i < n; i++) {
					int r = (offset + i) / mb.getNumColumns();
					int c = (offset + i) % mb.getNumColumns();
					ib.put((int) mb.getDouble(r, c));
				}
				return n * 4;
			};
			case UINT8 -> (src, offset, bb) -> {
				MatrixBlock mb = (MatrixBlock) src;
				int n = Math.min(bb.limit(), mb.getNumRows() * mb.getNumColumns() - offset);
				for (int i = 0; i < n; i++) {
					int r = (offset + i) / mb.getNumColumns();
					int c = (offset + i) % mb.getNumColumns();
					bb.put(i, (byte) ((int) mb.getDouble(r, c) & 0xFF));
				}
				return n;
			};
			default -> throw new UnsupportedOperationException("Unsupported type: " + type);
		};
	}

	/**
	 * Symmetric with readNumpyArrayInBatches — writes data in batches with handshake.
	 */
	public static long writeNumpyArrayInBatches(BufferedOutputStream out, int id, int batchSize,
												int numElem, Types.ValueType type, MatrixBlock mb)
			throws IOException {
		int elemSize = getElementSize(type);
		long totalBytesWritten = 0;

		try {
			writeHandshake(id, out);
			long bytesRemaining = ((long) numElem) * elemSize;
			byte[] buffer = new byte[batchSize];
			BufferWriter writer = getBufferWriter(type);

			int offset = 0;
			while (bytesRemaining > 0) {
				int chunk = (int) Math.min(batchSize, bytesRemaining);
				ByteBuffer bb = newLittleEndianBuffer(buffer, chunk);
				int bytesFilled = writer.writeFrom(mb, offset, bb);
				out.write(buffer, 0, bytesFilled);
				totalBytesWritten += bytesFilled;
				bytesRemaining -= bytesFilled;
				offset += bytesFilled / elemSize;
			}

			out.flush();
			writeHandshake(id, out);
			return totalBytesWritten;
		} catch (Exception e) {
			LOG.error("Error occurred while writing data to pipe #" + id, e);
			throw e;
		}
	}

	public static Array<?> readFrameColumnFromPipe(
			BufferedInputStream in, int id, int rows, int totalBytes, int batchSize,
			Types.ValueType type) throws IOException {

		long tStart = System.nanoTime();
		long tIoStart, tIoTotal = 0;
		long tDecodeTotal = 0;
		int numStrings = 0;
		
		readHandshake(id, in);
		Array<?> array = ArrayFactory.allocate(type, rows);
		byte[] buffer = new byte[batchSize];
		try {
			if (type != Types.ValueType.STRING) {
				tIoStart = System.nanoTime();
				readFixedTypeColumn(in, array, type, rows, totalBytes, buffer);
				tIoTotal = System.nanoTime() - tIoStart;
				readHandshake(id, in);
			} else {
				tIoStart = System.nanoTime();
				VarFillTiming timing = readVariableTypeColumn(in, id, array, type, rows, buffer);
				tIoTotal = System.nanoTime() - tIoStart;
				tDecodeTotal = timing.decodeTime;
				numStrings = timing.numStrings;
			}
		} catch (Exception e) {
			LOG.error("Error occurred while reading FrameBlock column from pipe #" + id, e);
			throw e;
		}
		
		long tTotal = System.nanoTime() - tStart;
		if (type == Types.ValueType.STRING) {
			LOG.debug(String.format(
				"Java readFrameColumnFromPipe timing: total=%.3fs, I/O=%.3fs (%.1f%%), decode=%.3fs (%.1f%%), strings=%d",
				tTotal / 1e9, tIoTotal / 1e9, 100.0 * tIoTotal / tTotal,
				tDecodeTotal / 1e9, 100.0 * tDecodeTotal / tTotal, numStrings));
		}
		return array;
	}
	
	private static class VarFillTiming {
		long decodeTime;
		int numStrings;
		VarFillTiming(long decodeTime, int numStrings) {
			this.decodeTime = decodeTime;
			this.numStrings = numStrings;
		}
	}

	private static void readFixedTypeColumn(
			BufferedInputStream in, Array<?> array,
			Types.ValueType type, int rows, int totalBytes, byte[] buffer) throws IOException {

		int elemSize = getElementSize(type);
		int expected = rows * elemSize;
		if (totalBytes != expected)
			throw new IOException("Expected " + expected + " bytes but got " + totalBytes);

		int offset = 0;
		long bytesRemaining = totalBytes;

		while (bytesRemaining > 0) {
			int chunk = (int) Math.min(buffer.length, bytesRemaining);
			readFully(in, buffer, chunk);
			offset = fillFixedArrayFromBytes(array, type, offset, buffer, chunk);
			bytesRemaining -= chunk;
		}
	}

	private static int fillFixedArrayFromBytes(
			Array<?> array, Types.ValueType type, int offsetOut,
			byte[] buffer, int currentBatchSize) {

		ByteBuffer bb = newLittleEndianBuffer(buffer, currentBatchSize);

		switch (type) {
			case FP64 -> {
				DoubleBuffer db = bb.asDoubleBuffer();
				while (db.hasRemaining())
					array.set(offsetOut++, db.get());
			}
			case FP32 -> {
				FloatBuffer fb = bb.asFloatBuffer();
				while (fb.hasRemaining())
					array.set(offsetOut++, fb.get());
			}
			case INT64 -> {
				LongBuffer lb = bb.asLongBuffer();
				while (lb.hasRemaining())
					array.set(offsetOut++, lb.get());
			}
			case INT32 -> {
				IntBuffer ib = bb.asIntBuffer();
				while (ib.hasRemaining())
					array.set(offsetOut++, ib.get());
			}
			case UINT8 -> {
				for (int i = 0; i < currentBatchSize; i++)
					array.set(offsetOut++, (int) (bb.get(i) & 0xFF));
			}
			case BOOLEAN -> {
				for (int i = 0; i < currentBatchSize; i++)
					array.set(offsetOut++, bb.get(i) != 0 ? 1.0 : 0.0);
			}
			default -> throw new UnsupportedOperationException("Unsupported fixed type: " + type);
		}
		return offsetOut;
	}

	private static VarFillTiming readVariableTypeColumn(
			BufferedInputStream in, int id, Array<?> array,
			Types.ValueType type, int elems, byte[] buffer) throws IOException {

		long tDecodeTotal = 0;
		int numStrings = 0;
		
		int offset = 0;
		// Use a reusable growable byte array to avoid repeated toByteArray() allocations
		byte[] combined = new byte[32 * 1024]; // Start with 32KB
		int combinedLen = 0;

		// Keep reading until all expected elements are filled
		while (offset < elems) {
			int chunk = in.read(buffer);

			// Ensure combined array is large enough
			if (combinedLen + chunk > combined.length) {
				// Grow array (double size, but at least accommodate new data)
				int newSize = Math.max(combined.length * 2, combinedLen + chunk);
				byte[] newCombined = new byte[newSize];
				System.arraycopy(combined, 0, newCombined, 0, combinedLen);
				combined = newCombined;
			}
			
			// Append newly read bytes
			System.arraycopy(buffer, 0, combined, combinedLen, chunk);
			combinedLen += chunk;

			// Try decoding as many complete elements as possible
			long tDecodeStart = System.nanoTime();
			VarFillResult res = fillVariableArrayFromBytes(array, offset, elems, combined, combinedLen, type);
			tDecodeTotal += System.nanoTime() - tDecodeStart;
			int stringsDecoded = res.offsetOut - offset;
			numStrings += stringsDecoded;
			offset = res.offsetOut;

			// Retain any incomplete trailing bytes by shifting them to the start
			int remainingBytes = res.remainingBytes;
			if (remainingBytes > 0) {
				// Move remaining bytes to the start of the buffer
				System.arraycopy(combined, combinedLen - remainingBytes, combined, 0, remainingBytes);
				combinedLen = remainingBytes;
			} else {
				combinedLen = 0;
			}
		}

		// ---- handshake check ----
		if(combinedLen == 0)
			readHandshake(id, in);
		else if (combinedLen == 4) {
			byte[] tail = new byte[4];
			System.arraycopy(combined, 0, tail, 0, 4);
			compareHandshakeIds(id, in, tail);
		}
		else
			throw new IOException("Expected 4-byte handshake after last element, found " + combinedLen + " bytes");

		return new VarFillTiming(tDecodeTotal, numStrings);
	}

	/**
	 * Result container for variable-length decoding.
	 *
	 * @param offsetOut		 number of elements written to the output array
	 * @param remainingBytes number of unconsumed tail bytes (partial element)
	 */
	private record VarFillResult(int offsetOut, int remainingBytes) {
	}

	private static VarFillResult fillVariableArrayFromBytes(
			Array<?> array, int offsetOut, int maxOffset, byte[] buffer,
			int currentBatchSize, Types.ValueType type) {

		ByteBuffer bb = newLittleEndianBuffer(buffer, currentBatchSize);
		int bytesConsumed = 0;

		// Each variable-length element = [int32 length][payload...]
		while (bb.remaining() >= 4 && offsetOut < maxOffset) {
			bb.mark();
			int len = bb.getInt();

			if (len < 0) {
				// null string
				array.set(offsetOut++, (String) null);
				bytesConsumed = bb.position();
				continue;
			}
			if (bb.remaining() < len) {
				// Not enough bytes for full payload → rollback and stop
				bb.reset();
				break;
			}
			

			switch (type) {
				case STRING -> {
					int stringStart = bb.position();
					
					byte[] backingArray = bb.array();
					int arrayOffset = bb.arrayOffset() + stringStart;
					String s = new String(backingArray, arrayOffset, len, StandardCharsets.UTF_8);
					array.set(offsetOut++, s);
					
					bb.position(stringStart + len);
				}

				default -> throw new UnsupportedOperationException(
						"Unsupported variable-length type: " + type);
			}

			bytesConsumed = bb.position();
		}

		int remainingBytes = currentBatchSize - bytesConsumed;
		return new VarFillResult(offsetOut, remainingBytes);
	}

	/**
	 * Symmetric with readFrameColumnFromPipe — writes FrameBlock column data to pipe.
	 * Supports both fixed-size types and variable-length types (strings).
	 */
	public static long writeFrameColumnToPipe(
			BufferedOutputStream out, int id, int batchSize,
			Array<?> array, Types.ValueType type) throws IOException {
		
		long tStart = System.nanoTime();
		long tIoStart, tIoTotal = 0;
		long tEncodeTotal = 0;
		int numStrings = 0;
		long totalBytesWritten = 0;

		try {
			writeHandshake(id, out);
			
			if (type != Types.ValueType.STRING) {
				tIoStart = System.nanoTime();
				totalBytesWritten = writeFixedTypeColumn(out, array, type, batchSize);
				tIoTotal = System.nanoTime() - tIoStart;
			} else {
				tIoStart = System.nanoTime();
				VarWriteTiming timing = writeVariableTypeColumn(out, array, type, batchSize);
				tIoTotal = System.nanoTime() - tIoStart;
				tEncodeTotal = timing.encodeTime;
				numStrings = timing.numStrings;
				totalBytesWritten = timing.totalBytes;
			}
			
			out.flush();
			writeHandshake(id, out);
			
			long tTotal = System.nanoTime() - tStart;
			if (type == Types.ValueType.STRING) {
				LOG.debug(String.format(
					"Java writeFrameColumnToPipe timing: total=%.3fs, I/O=%.3fs (%.1f%%), encode=%.3fs (%.1f%%), strings=%d",
					tTotal / 1e9, tIoTotal / 1e9, 100.0 * tIoTotal / tTotal,
					tEncodeTotal / 1e9, 100.0 * tEncodeTotal / tTotal, numStrings));
			}
			
			return totalBytesWritten;
		} catch (Exception e) {
			LOG.error("Error occurred while writing FrameBlock column to pipe #" + id, e);
			throw e;
		}
	}

	private static class VarWriteTiming {
		long encodeTime;
		int numStrings;
		long totalBytes;
		VarWriteTiming(long encodeTime, int numStrings, long totalBytes) {
			this.encodeTime = encodeTime;
			this.numStrings = numStrings;
			this.totalBytes = totalBytes;
		}
	}

	private static long writeFixedTypeColumn(
			BufferedOutputStream out, Array<?> array,
			Types.ValueType type, int batchSize) throws IOException {
		
		int elemSize = getElementSize(type);
		int rows = array.size();
		long totalBytes = (long) rows * elemSize;
		
		byte[] buffer = new byte[batchSize];
		int arrayIndex = 0;
		int bufferPos = 0;

		while (arrayIndex < rows) {
			// Calculate how many elements can fit in the remaining buffer space
			int remainingBufferSpace = batchSize - bufferPos;
			int elementsToWrite = Math.min((remainingBufferSpace / elemSize), rows - arrayIndex);
			
			if (elementsToWrite == 0) {
				// Buffer is full, flush it
				out.write(buffer, 0, bufferPos);
				bufferPos = 0;
				continue;
			}
			
			// Convert elements to bytes directly into the buffer
			ByteBuffer bb = ByteBuffer.wrap(buffer, bufferPos, elementsToWrite * elemSize)
					.order(ByteOrder.LITTLE_ENDIAN);
			
			switch (type) {
				case FP64 -> {
					DoubleBuffer db = bb.asDoubleBuffer();
					for (int i = 0; i < elementsToWrite; i++) {
						db.put(array.getAsDouble(arrayIndex++));
					}
					bufferPos += elementsToWrite * 8;
				}
				case FP32 -> {
					FloatBuffer fb = bb.asFloatBuffer();
					for (int i = 0; i < elementsToWrite; i++) {
						fb.put((float) array.getAsDouble(arrayIndex++));
					}
					bufferPos += elementsToWrite * 4;
				}
				case INT64 -> {
					LongBuffer lb = bb.asLongBuffer();
					for (int i = 0; i < elementsToWrite; i++) {
						lb.put((long) array.getAsDouble(arrayIndex++));
					}
					bufferPos += elementsToWrite * 8;
				}
			case INT32 -> {
				IntBuffer ib = bb.asIntBuffer();
				for (int i = 0; i < elementsToWrite; i++) {
					ib.put((int) array.getAsDouble(arrayIndex++));
				}
				bufferPos += elementsToWrite * 4;
			}
			case BOOLEAN -> {
				for (int i = 0; i < elementsToWrite; i++) {
					buffer[bufferPos++] = (byte) (array.getAsDouble(arrayIndex++) != 0.0 ? 1 : 0);
				}
			}
			default -> throw new UnsupportedOperationException("Unsupported type: " + type);
			}
		}

		out.write(buffer, 0, bufferPos);
		return totalBytes;
	}

	private static VarWriteTiming writeVariableTypeColumn(
			BufferedOutputStream out, Array<?> array,
			Types.ValueType type, int batchSize) throws IOException {
		
		long tEncodeTotal = 0;
		int numStrings = 0;
		long totalBytesWritten = 0;
		
		byte[] buffer = new byte[batchSize]; // Use 2x batch size like Python side
		int pos = 0;

		int rows = array.size();
		
		for (int i = 0; i < rows; i++) {
			numStrings++;
			
			// Get string value
			Object value = array.get(i);
			boolean isNull = (value == null);
			
			int length;
			byte[] encoded;
			
			if (isNull) {
				// Use -1 as marker for null values
				length = -1;
				encoded = new byte[0];
			} else {
				// Encode to UTF-8
				long tEncodeStart = System.nanoTime();
				String str = value.toString();
				encoded = str.getBytes(StandardCharsets.UTF_8);
				tEncodeTotal += System.nanoTime() - tEncodeStart;
				length = encoded.length;
			}
			
			int entrySize = 4 + (length >= 0 ? length : 0); // length prefix + data (or just prefix for null)

			// If next string doesn't fit comfortably, flush first half
			if (pos + entrySize > batchSize) {
				out.write(buffer, 0, pos);
				totalBytesWritten += pos;
				pos = 0;
			}

			// Write length prefix (little-endian) - use -1 for null
			ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
			bb.putInt(length);
			System.arraycopy(bb.array(), 0, buffer, pos, 4);
			pos += 4;
			
			// Write the encoded bytes (skip for null)
			if (length > 0) {
				int remainingBytes = length;
				int encodedOffset = 0;
				while (remainingBytes > 0) {
					int chunk = Math.min(remainingBytes, batchSize - pos);
					System.arraycopy(encoded, encodedOffset, buffer, pos, chunk);
					pos += chunk;
					if (pos == batchSize) {
						out.write(buffer, 0, pos);
						totalBytesWritten += pos;
						pos = 0;
					}
					encodedOffset += chunk;
					remainingBytes -= chunk;
				}
			}
		}

		// Flush the tail
		if (pos > 0) {
			out.write(buffer, 0, pos);
			totalBytesWritten += pos;
		}

		return new VarWriteTiming(tEncodeTotal, numStrings, totalBytesWritten);
	}
}