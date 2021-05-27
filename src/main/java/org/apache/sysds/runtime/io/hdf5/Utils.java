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


package org.apache.sysds.runtime.io.hdf5;

import org.apache.commons.lang3.ArrayUtils;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.BitSet;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public final class Utils {
	private static final CharsetEncoder ASCII = StandardCharsets.US_ASCII.newEncoder();

	private Utils() {
		throw new AssertionError("No instances of Utils");
	}

	/**
	 * Converts an address to a hex string for display
	 *
	 * @param address to convert to Hex
	 * @return the address as a hex string
	 */
	public static String toHex(long address) {
		if (address == Constants.UNDEFINED_ADDRESS) {
			return "UNDEFINED";
		}
		return "0x" + Long.toHexString(address);
	}

	/**
	 * Reads ASCII string from the buffer until a null character is reached. This
	 * will read from the buffers current position. After the method the buffer
	 * position will be after the null character.
	 *
	 * @param buffer to read from
	 * @return the string read from the buffer
	 * @throws IllegalArgumentException if the end of the buffer if reached before
	 *                                  and null terminator
	 */
	public static String readUntilNull(ByteBuffer buffer) {
		StringBuilder sb = new StringBuilder(buffer.remaining());
		while (buffer.hasRemaining()) {
			byte b = buffer.get();
			if (b == Constants.NULL) {
				return sb.toString();
			}
			sb.append((char) b);
		}
		throw new IllegalArgumentException("End of buffer reached before NULL");
	}

	/**
	 * Check the provided name to see if it is valid for a HDF5 identifier. Checks
	 * name only contains ASCII characters and does not contain '/' or '.' which are
	 * reserved characters.
	 *
	 * @param name To check if valid
	 * @return <code>true</code> if this is a valid HDF5 name, <code>false</code>
	 *         otherwise
	 */
	public static boolean validateName(String name) {
		return ASCII.canEncode(name) && !name.contains("/") && !name.contains(".");
	}

	/**
	 * Moves the position of the {@link ByteBuffer} to the next position aligned on
	 * 8 bytes. If the buffer position is already a multiple of 8 the position will
	 * not be changed.
	 *
	 * @param bb the buffer to be aligned
	 */
	public static void seekBufferToNextMultipleOfEight(ByteBuffer bb) {
		int pos = bb.position();
		if (pos % 8 == 0) {
			return; // Already on a 8 byte multiple
		}
		bb.position(pos + (8 - (pos % 8)));
	}

	/**
	 * This reads the requested number of bytes from the buffer and returns the data
	 * as an unsigned <code>int</code>. After this call the buffer position will be
	 * advanced by the specified length.
	 * <p>
	 * This is used in HDF5 to read "size of lengths" and "size of offsets"
	 *
	 * @param buffer to read from
	 * @param length the number of bytes to read
	 * @return the <code>int</code> value read from the buffer
	 * @throws ArithmeticException      if the data cannot be safely converted to an
	 *                                  unsigned <code>int</code>
	 * @throws IllegalArgumentException if the length requested is not supported i.e
	 *                                  &gt; 8
	 */
	public static int readBytesAsUnsignedInt(ByteBuffer buffer, int length) {
		switch (length) {
		case 1:
			return Byte.toUnsignedInt(buffer.get());
		case 2:
			return Short.toUnsignedInt(buffer.getShort());
		case 3:
			return readArbitraryLengthBytesAsUnsignedInt(buffer, length);
		case 4:
			int value = buffer.getInt();
			if (value < 0) {
				throw new ArithmeticException("Could not convert to unsigned");
			}
			return value;
		case 5:
		case 6:
		case 7:
			return readArbitraryLengthBytesAsUnsignedInt(buffer, length);
		case 8:
			// Throws if the long can't be converted safely
			return Math.toIntExact(buffer.getLong());
		default:
			throw new IllegalArgumentException("Couldn't read " + length + " bytes as int");
		}
	}

	/**
	 * This method is used when the length required is awkward i.e. no support
	 * directly from {@link ByteBuffer}
	 *
	 * @param buffer to read from
	 * @param length the number of bytes to read
	 * @return the long value read from the buffer
	 * @throws ArithmeticException if the data cannot be safely converted to an
	 *                             unsigned long
	 */
	private static int readArbitraryLengthBytesAsUnsignedInt(ByteBuffer buffer, int length) {
		// Here we will use BigInteger to convert a byte array
		byte[] bytes = new byte[length];
		buffer.get(bytes);
		// BigInteger needs big endian so flip the order if needed
		if (buffer.order() == LITTLE_ENDIAN) {
			ArrayUtils.reverse(bytes);
		}
		// Convert to a unsigned long throws if it overflows
		return new BigInteger(1, bytes).intValueExact();
	}

	/**
	 * This reads the requested number of bytes from the buffer and returns the data
	 * as an unsigned long. After this call the buffer position will be advanced by
	 * the specified length.
	 * <p>
	 * This is used in HDF5 to read "size of lengths" and "size of offsets"
	 *
	 * @param buffer to read from
	 * @param length the number of bytes to read
	 * @return the long value read from the buffer
	 * @throws ArithmeticException      if the data cannot be safely converted to an
	 *                                  unsigned long
	 * @throws IllegalArgumentException if the length requested is not supported;
	 */
	public static long readBytesAsUnsignedLong(ByteBuffer buffer, int length) {
		switch (length) {
		case 1:
			return Byte.toUnsignedLong(buffer.get());
		case 2:
			return Short.toUnsignedLong(buffer.getShort());
		case 3:
			return readArbitraryLengthBytesAsUnsignedLong(buffer, length);
		case 4:
			return Integer.toUnsignedLong(buffer.getInt());
		case 5:
		case 6:
		case 7:
			return readArbitraryLengthBytesAsUnsignedLong(buffer, length);
		case 8:
			long value = buffer.getLong();
			if (value < 0 && value != Constants.UNDEFINED_ADDRESS) {
				throw new ArithmeticException("Could not convert to unsigned");
			}
			return value;
		default:
			throw new IllegalArgumentException("Couldn't read " + length + " bytes as int");
		}
	}

	/**
	 * This method is used when the length required is awkward i.e. no support
	 * directly from {@link ByteBuffer}
	 *
	 * @param buffer to read from
	 * @param length the number of bytes to read
	 * @return the long value read from the buffer
	 * @throws ArithmeticException if the data cannot be safely converted to an
	 *                             unsigned long
	 */
	private static long readArbitraryLengthBytesAsUnsignedLong(ByteBuffer buffer, int length) {
		// Here we will use BigInteger to convert a byte array
		byte[] bytes = new byte[length];
		buffer.get(bytes);
		// BigInteger needs big endian so flip the order if needed
		if (buffer.order() == LITTLE_ENDIAN) {
			ArrayUtils.reverse(bytes);
		}
		// Convert to a unsigned long throws if it overflows
		return new BigInteger(1, bytes).longValueExact();
	}

	/**
	 * Creates a new {@link ByteBuffer} of the specified length. The new buffer will
	 * start at the current position of the source buffer and will be of the
	 * specified length. The {@link ByteOrder} of the new buffer will be the same as
	 * the source buffer. After the call the source buffer position will be
	 * incremented by the length of the sub-buffer. The new buffer will share the
	 * backing data with the source buffer.
	 *
	 * @param source the buffer to take the sub buffer from
	 * @param length the size of the new sub-buffer
	 * @return the new sub buffer
	 */
	public static ByteBuffer createSubBuffer(ByteBuffer source, int length) {
		ByteBuffer headerData = source.slice();
		headerData.limit(length);
		headerData.order(source.order());

		// Move the buffer past this header
		source.position(source.position() + length);
		return headerData;
	}

	private static final BigInteger TWO = BigInteger.valueOf(2);

	/**
	 * Takes a {@link BitSet} and a range of bits to inspect and converts the bits
	 * to a integer.
	 *
	 * @param bits   to inspect
	 * @param start  the first bit
	 * @param length the number of bits to inspect
	 * @return the integer represented by the provided bits
	 */
	public static int bitsToInt(BitSet bits, int start, int length) {
		if (length <= 0) {
			throw new IllegalArgumentException("length must be >0");
		}
		BigInteger result = BigInteger.ZERO;
		for (int i = 0; i < length; i++) {
			if (bits.get(start + i)) {
				result = result.add(TWO.pow(i));
			}
		}
		return result.intValue();
	}

	/**
	 * Calculates how many bytes are needed to store the given unsigned number.
	 *
	 * @param number to store
	 * @return the number of bytes needed to hold this number
	 * @throws IllegalArgumentException if a negative number is given
	 */
	public static int bytesNeededToHoldNumber(long number) {
		if (number < 0) {
			throw new IllegalArgumentException("Only for unsigned numbers");
		}
		if (number == 0) {
			return 1;
		}
		return (int) Math.ceil(BigInteger.valueOf(number).bitLength() / 8.0);
	}

    public static int[] linearIndexToDimensionIndex(int index, int[] dimensions) {
        int[] dimIndex = new int[dimensions.length];

        for (int i = dimIndex.length - 1; i >= 0; i--) {
            dimIndex[i] = index % dimensions[i];
            index = index / dimensions[i];
        }
        return dimIndex;
    }

    public static int dimensionIndexToLinearIndex(int[] index, int[] dimensions) {
        int linear = 0;
        for (int i = 0; i < dimensions.length; i++) {
            int temp = index[i];
            for (int j = i + 1; j < dimensions.length; j++) {
                temp *= dimensions[j];
            }
            linear += temp;
        }
        return linear;
    }

	/**
	 * Calculates the chunk offset from a given chunk index
	 *
	 * @param chunkIndex The index to calculate for
	 * @param chunkDimensions The chunk dimensions
	 * @param datasetDimensions The dataset dimensions
	 * @return The chunk offset for the chunk of this index
	 */
    public static int[] chunkIndexToChunkOffset(int chunkIndex, int[] chunkDimensions, int[] datasetDimensions) {
		final int[] chunkOffset = new int[chunkDimensions.length];

		// Start from the slowest dim
		for (int i = 0; i < chunkOffset.length; i++) {
			// Find out how many chunks make one chunk in this dim
			int chunksBelowThisDim = 1;
			// Start one dim faster
			for (int j = i+ 1; j < chunkOffset.length; j++) {
				chunksBelowThisDim *= (int) Math.ceil((double) datasetDimensions[j] / chunkDimensions[j]);
			}

			chunkOffset[i] = (chunkIndex / chunksBelowThisDim ) * chunkDimensions[i];
			chunkIndex -= chunkOffset[i]/chunkDimensions[i] * chunksBelowThisDim;
		}

		return chunkOffset;
	}

	/**
	 * Removes the zeroth (leading) index. e.g [1,2,3] → [2,3]
	 *
	 * @param dims the array to strip
	 * @return dims with the zeroth element removed
	 */
	public static int[] stripLeadingIndex(int[] dims) {
		return Arrays.copyOfRange(dims, 1, dims.length);
	}
}
