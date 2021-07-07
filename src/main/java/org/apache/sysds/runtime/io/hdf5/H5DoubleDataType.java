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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.BitSet;

public class H5DoubleDataType {

	private int version;
	private int dataClass;
	private int size;
	protected BitSet classBits;

	private ByteOrder order;
	private boolean lowPadding;
	private boolean highPadding;
	private boolean internalPadding;
	private int mantissaNormalization;
	private int signLocation;

	// Properties
	private short bitOffset;
	private short bitPrecision;
	private byte exponentLocation;
	private byte exponentSize;
	private byte mantissaLocation;
	private byte mantissaSize;
	private int exponentBias;

	public H5DoubleDataType() {
	}

	public H5DoubleDataType(ByteBuffer bb) {

		// Class and version
		final BitSet classAndVersion = BitSet.valueOf(new byte[] {bb.get()});
		dataClass = Utils.bitsToInt(classAndVersion, 0, 4);
		version = Utils.bitsToInt(classAndVersion, 4, 4);

		byte[] classBytes = new byte[3];
		bb.get(classBytes);

		classBits = BitSet.valueOf(classBytes);
		// Size
		size = Utils.readBytesAsUnsignedInt(bb, 4);

		if(classBits.get(6)) {
			throw new H5RuntimeException("VAX endian is not supported");
		}
		if(classBits.get(0)) {
			order = ByteOrder.BIG_ENDIAN;
		}
		else {
			order = ByteOrder.LITTLE_ENDIAN;
		}

		lowPadding = classBits.get(1);
		highPadding = classBits.get(2);
		internalPadding = classBits.get(3);

		// Mask the 4+5 bits and shift to the end
		mantissaNormalization = Utils.bitsToInt(classBits, 4, 2);

		signLocation = Utils.bitsToInt(classBits, 8, 8);

		// Properties
		bitOffset = bb.getShort();
		bitPrecision = bb.getShort();
		exponentLocation = bb.get();
		exponentSize = bb.get();
		mantissaLocation = bb.get();
		mantissaSize = bb.get();
		exponentBias = bb.getInt();
	}

	public void toBuffer(H5BufferBuilder bb) {
		byte classAndVersion = 17;
		bb.writeByte(classAndVersion);

		byte[] classBytes = {32, 63, 0};
		bb.writeBytes(classBytes);

		bb.writeInt(8);

		//bitOffset
		bb.writeShort((short) 0);

		//bitPrecision
		bb.writeShort((short) 64);

		//exponentLocation
		bb.writeByte(52);

		//exponentSize
		bb.writeByte(11);

		//mantissaLocation
		bb.writeByte(0);

		//mantissaSize
		bb.writeByte(52);

		//exponentBias
		bb.writeInt(1023);

		// reserved
		bb.writeInt(0);
	}

	public void fillData(ByteBuffer buffer, double[] data) {
		DoubleBuffer db = buffer.asDoubleBuffer();
		db.get(data);
	}

	public int getVersion() {
		return version;
	}

	public int getDataClass() {
		return dataClass;
	}

	public int getSize() {
		return size;
	}

	public BitSet getClassBits() {
		return classBits;
	}

	public ByteOrder getOrder() {
		return order;
	}

	public boolean isLowPadding() {
		return lowPadding;
	}

	public boolean isHighPadding() {
		return highPadding;
	}

	public boolean isInternalPadding() {
		return internalPadding;
	}

	public int getMantissaNormalization() {
		return mantissaNormalization;
	}

	public int getSignLocation() {
		return signLocation;
	}

	public short getBitOffset() {
		return bitOffset;
	}

	public short getBitPrecision() {
		return bitPrecision;
	}

	public byte getExponentLocation() {
		return exponentLocation;
	}

	public byte getExponentSize() {
		return exponentSize;
	}

	public byte getMantissaLocation() {
		return mantissaLocation;
	}

	public byte getMantissaSize() {
		return mantissaSize;
	}

	public int getExponentBias() {
		return exponentBias;
	}
}
