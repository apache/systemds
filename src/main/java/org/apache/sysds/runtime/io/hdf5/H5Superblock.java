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
import java.util.Arrays;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5Superblock {

	protected static final byte[] HDF5_FILE_SIGNATURE = new byte[] {(byte) 137, 72, 68, 70, 13, 10, 26, 10};
	protected static final int HDF5_FILE_SIGNATURE_LENGTH = HDF5_FILE_SIGNATURE.length;
	public int versionOfSuperblock;
	public int versionNumberOfTheFileFreeSpaceInformation;
	public int versionOfRootGroupSymbolTableEntry;
	public int versionOfSharedHeaderMessageFormat;
	public int sizeOfOffsets;
	public int sizeOfLengths;
	public int groupLeafNodeK;
	public int groupInternalNodeK;
	public long baseAddressByte;
	public long addressOfGlobalFreeSpaceIndex;
	public long endOfFileAddress;
	public long driverInformationBlockAddress;
	public long rootGroupSymbolTableAddress;

	public H5Superblock() {
	}

	static boolean verifySignature(H5ByteReader reader, long offset) {
		try {
			ByteBuffer signature = reader.read(offset, HDF5_FILE_SIGNATURE_LENGTH);
			byte[] sigBytes = new byte[HDF5_FILE_SIGNATURE_LENGTH];
			signature.get(sigBytes);
			return Arrays.equals(HDF5_FILE_SIGNATURE, sigBytes);
		}
		catch(Exception e) {
			throw new H5RuntimeException("Failed to read from address: " + offset, e);
		}
	}

	public H5Superblock(H5ByteReader reader, long address) {

		// Calculated bytes for the super block header is = 56
		int superBlockHeaderSize = 12;

		long cursor = address + HDF5_FILE_SIGNATURE_LENGTH;

		try {
			ByteBuffer header = reader.read(cursor, superBlockHeaderSize);
			header.order(LITTLE_ENDIAN);
			header.rewind();
			cursor += superBlockHeaderSize;

			// Version # of Superblock
			versionOfSuperblock = header.get();

			if(versionOfSuperblock != 0 && versionOfSuperblock != 1) {
				throw new H5RuntimeException("Detected superblock version not 0 or 1");
			}

			// Version # of File Free-space Storage
			versionNumberOfTheFileFreeSpaceInformation = header.get();

			// Version # of Root Group Symbol Table Entry
			versionOfRootGroupSymbolTableEntry = header.get();

			// Skip reserved byte
			header.position(header.position() + 1);

			// Version # of Shared Header Message Format
			versionOfSharedHeaderMessageFormat = header.get();

			// Size of Offsets
			sizeOfOffsets = Byte.toUnsignedInt(header.get());

			// Size of Lengths
			sizeOfLengths = Byte.toUnsignedInt(header.get());

			// Skip reserved byte
			header.position(header.position() + 1);

			// Group Leaf Node K
			groupLeafNodeK = Short.toUnsignedInt(header.getShort());

			// Group Internal Node K
			groupInternalNodeK = Short.toUnsignedInt(header.getShort());

			// File Consistency Flags (skip)
			cursor += 4;

			int nextSectionSize = 4 * sizeOfOffsets;
			header = reader.read(cursor, nextSectionSize);
			header.order(LITTLE_ENDIAN);
			header.rewind();
			cursor += nextSectionSize;

			// Base Address
			baseAddressByte = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

			// Address of Global Free-space Index
			addressOfGlobalFreeSpaceIndex = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

			// End of File Address
			endOfFileAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

			// Driver Information Block Address
			driverInformationBlockAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

			// Root Group Symbol Table Entry Address
			rootGroupSymbolTableAddress = cursor;
		}
		catch(Exception e) {
			throw new H5RuntimeException("Failed to read superblock from address " + address, e);
		}
	}

	protected H5BufferBuilder toBuffer() {
		H5BufferBuilder bb = new H5BufferBuilder();
		this.toBuffer(bb);
		return bb;
	}

	public void toBuffer(H5BufferBuilder bb) {

		// HDF5 File Signature (8 bytes)
		bb.writeBytes(HDF5_FILE_SIGNATURE);

		// Version # of Superblock
		bb.writeByte(versionOfSuperblock);

		// Version # of File Free-space Storage
		bb.writeByte(versionNumberOfTheFileFreeSpaceInformation);

		// Version # of Root Group Symbol Table Entry
		bb.writeByte(versionOfRootGroupSymbolTableEntry);

		// Skip reserved byte
		bb.writeByte(0);

		// Version # of Shared Header Message Format
		bb.writeByte(versionOfSharedHeaderMessageFormat);

		// Size of Offsets
		bb.writeByte(sizeOfOffsets);

		// Size of Lengths
		bb.writeByte(sizeOfLengths);

		// Skip reserved byte
		bb.writeByte(0);

		// Group Leaf Node K
		bb.writeShort((short) groupLeafNodeK);

		// Group Internal Node K
		bb.writeShort((short) groupInternalNodeK);

		// File Consistency Flags (skip)
		bb.writeInt(0);

		// Base Address
		bb.writeLong(baseAddressByte);

		// Address of Global Free-space Index
		bb.writeLong(addressOfGlobalFreeSpaceIndex);

		// End of File Address
		bb.writeLong(endOfFileAddress);

		// Driver Information Block Address
		bb.writeLong(driverInformationBlockAddress);
	}
}
