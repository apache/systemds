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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5BufferBuilder {

	private final ByteArrayOutputStream byteArrayOutputStream;
	private final DataOutputStream dataOutputStream;
	private ByteOrder byteOrder = LITTLE_ENDIAN;

	public H5BufferBuilder() {
		this.byteArrayOutputStream = new ByteArrayOutputStream();
		this.dataOutputStream = new DataOutputStream(byteArrayOutputStream);
	}

	public int getSize() {
		return dataOutputStream.size();
	}

	public void writeByte(int i) {
		try {
			dataOutputStream.writeByte(i);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void writeBytes(byte[] bytes) {
		try {
			dataOutputStream.write(bytes);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void writeInt(int i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Integer.reverseBytes(i);
			}
			dataOutputStream.writeInt(i);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void writeShort(short i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Short.reverseBytes(i);
			}
			dataOutputStream.writeShort(i);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void writeLong(long l) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				l = Long.reverseBytes(l);
			}
			dataOutputStream.writeLong(l);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public Double getNewVal(double d) {
		return d;
	}

	public static BitSet convert(long value) {
		BitSet bits = new BitSet();
		String a = "";
		int index = 0;
		while(value != 0L) {
			if(value % 2L != 0) {
				bits.set(index);
				a += "1";
			}
			else
				a += "0";
			++index;
			value = value >>> 1;
		}
		System.out.println(a);
		return bits;
	}

	public void writeDouble(double l) {
		try {
			this.writeLong(Double.doubleToLongBits(l));
		}
		catch(Exception e) {
			throw new H5RuntimeException(e);
		}
	}

	public void write(long v, int sizeOfLength) {
		try {
			switch(sizeOfLength) {
				case 2:
					this.writeShort((short) v);
					break;
				case 4:
					this.writeInt((int) v);
					break;
				case 8:
					this.writeLong(v);
					break;

			}
		}
		catch(Exception e) {
			throw new H5RuntimeException(e);
		}

	}

	public ByteBuffer build() {
		try {
			ByteBuffer byteBuffer = ByteBuffer.wrap(byteArrayOutputStream.toByteArray());
			byteBuffer.order(byteOrder);
			dataOutputStream.close();
			byteArrayOutputStream.close();
			return byteBuffer;
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public ByteBuffer noOrderBuild() {
		try {
			ByteBuffer byteBuffer = ByteBuffer.wrap(byteArrayOutputStream.toByteArray());
			dataOutputStream.close();
			byteArrayOutputStream.close();
			return byteBuffer;
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void writeBitSet(BitSet bitSet, int length) {
		if(bitSet.length() > length) {
			throw new H5RuntimeException("BitSet is longer than length provided");
		}
		try {
			final byte[] bytes = Arrays.copyOf(bitSet.toByteArray(), length); // Ensure empty Bitset are not shortened
			dataOutputStream.write(bytes);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void goToPositionWithWriteZero(long pos) {
		long gap = pos - this.dataOutputStream.size();
		byte[] gapByte = new byte[(int) gap];
		this.writeBytes(gapByte);
	}

}
