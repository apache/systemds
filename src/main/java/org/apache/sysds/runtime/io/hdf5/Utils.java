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
import java.util.BitSet;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public final class Utils {

	private Utils() {
		throw new AssertionError("No instances of Utils");
	}

	private static final BigInteger TWO = BigInteger.valueOf(2);

	public static String readUntilNull(ByteBuffer buffer) {
		StringBuilder sb = new StringBuilder(buffer.remaining());
		while(buffer.hasRemaining()) {
			byte b = buffer.get();
			if(b == H5Constants.NULL) {
				return sb.toString();
			}
			sb.append((char) b);
		}
		throw new IllegalArgumentException("End of buffer reached before NULL");
	}

	public static void seekBufferToNextMultipleOfEight(ByteBuffer bb) {
		int pos = bb.position();
		if(pos % 8 == 0) {
			return; // Already on a 8 byte multiple
		}
		bb.position(pos + (8 - (pos % 8)));
	}

	public static int readBytesAsUnsignedInt(ByteBuffer buffer, int length) {
		switch(length) {
			case 1:
				return Byte.toUnsignedInt(buffer.get());
			case 2:
				return Short.toUnsignedInt(buffer.getShort());
			case 3:
				return readArbitraryLengthBytesAsUnsignedInt(buffer, length);
			case 4:
				int value = buffer.getInt();
				if(value < 0) {
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

	private static int readArbitraryLengthBytesAsUnsignedInt(ByteBuffer buffer, int length) {
		byte[] bytes = new byte[length];
		buffer.get(bytes);
		if(buffer.order() == LITTLE_ENDIAN) {
			ArrayUtils.reverse(bytes);
		}
		return new BigInteger(1, bytes).intValueExact();
	}

	public static long readBytesAsUnsignedLong(ByteBuffer buffer, int length) {
		switch(length) {
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
				if(value < 0 && value != H5Constants.UNDEFINED_ADDRESS) {
					throw new ArithmeticException("Could not convert to unsigned");
				}
				return value;
			default:
				throw new IllegalArgumentException("Couldn't read " + length + " bytes as int");
		}
	}

	private static long readArbitraryLengthBytesAsUnsignedLong(ByteBuffer buffer, int length) {
		byte[] bytes = new byte[length];
		buffer.get(bytes);
		if(buffer.order() == LITTLE_ENDIAN) {
			ArrayUtils.reverse(bytes);
		}
		return new BigInteger(1, bytes).longValueExact();
	}

	public static ByteBuffer createSubBuffer(ByteBuffer source, int length) {
		ByteBuffer headerData = source.slice();
		headerData.limit(length);
		headerData.order(source.order());

		source.position(source.position() + length);
		return headerData;
	}

	public static int bitsToInt(BitSet bits, int start, int length) {
		if(length <= 0) {
			throw new IllegalArgumentException("length must be >0");
		}
		BigInteger result = BigInteger.ZERO;
		for(int i = 0; i < length; i++) {
			if(bits.get(start + i)) {
				result = result.add(TWO.pow(i));
			}
		}
		return result.intValue();
	}

}
