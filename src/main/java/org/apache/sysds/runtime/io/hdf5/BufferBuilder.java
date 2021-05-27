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

import org.apache.sysds.runtime.io.hdf5.checksum.ChecksumUtils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class BufferBuilder {

	private final ByteArrayOutputStream byteArrayOutputStream;
	private final DataOutputStream dataOutputStream; // Note always big endian
	private final ByteOrder byteOrder = LITTLE_ENDIAN;

	public BufferBuilder() {
		this.byteArrayOutputStream = new ByteArrayOutputStream();
		this.dataOutputStream = new DataOutputStream(byteArrayOutputStream);
	}

	public void printSize(){
		System.out.println("size= "+ this.dataOutputStream.size());
	}

	public void seekBufferBuilderToNextMultipleOfEight() {
		int pos = this.dataOutputStream.size();
		if (pos % 8 == 0) {
			return; // Already on a 8 byte multiple
		}
		for( int i =0; i<(8 - (pos % 8));i++){
			this.writeByte(0);
		}
	}

	public BufferBuilder writeByte(int i) {
		try {
			dataOutputStream.writeByte(i);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder writeBytes(byte[] bytes) {
		try {
			dataOutputStream.write(bytes);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder writeInt(int i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Integer.reverseBytes(i);
			}
			dataOutputStream.writeInt(i);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder writeShort(short i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Short.reverseBytes(i);
			}
			dataOutputStream.writeShort(i);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder writeLong(long l) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				l = Long.reverseBytes(l);
			}
			dataOutputStream.writeLong(l);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public ByteBuffer build() {
		try {
			ByteBuffer byteBuffer = ByteBuffer.wrap(byteArrayOutputStream.toByteArray());
			byteBuffer.order(byteOrder);
			dataOutputStream.close();
			byteArrayOutputStream.close();
			return byteBuffer;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder writeBitSet(BitSet bitSet, int length) {
		if(bitSet.length() > length) {
			throw new IllegalArgumentException("BitSet is longer than length provided");
		}
		try {
			final byte[] bytes = Arrays.copyOf(bitSet.toByteArray(), length); // Ensure empty Bitset are not shortened
			dataOutputStream.write(bytes);
			return this;
		} catch (IOException e) {
			throw new BufferBuilderException(e);
		}
	}

	public BufferBuilder goToPositionWithWriteZero(long pos){
		long gap= pos - this.dataOutputStream.size();
		byte[] gapByte=new byte[(int) gap];
		this.writeBytes(gapByte);
		return this;
	}

	public BufferBuilder appendChecksum() {
		writeInt(ChecksumUtils.checksum(byteArrayOutputStream.toByteArray()));
		return this;
	}

	public static final class BufferBuilderException extends HdfException {
		private BufferBuilderException(String message, Throwable throwable) {
			super(message, throwable);
		}

		private BufferBuilderException(Throwable throwable) {
			this("Error in BufferBuilder", throwable);
		}
	}

}
