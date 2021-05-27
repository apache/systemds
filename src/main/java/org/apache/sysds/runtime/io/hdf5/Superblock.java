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

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.hdf5.checksum.ChecksumUtils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

import static org.apache.sysds.runtime.io.hdf5.Utils.toHex;
import static java.nio.ByteOrder.LITTLE_ENDIAN;

public abstract class Superblock {

	private static final byte[] HDF5_FILE_SIGNATURE = new byte[] {(byte) 137, 72, 68, 70, 13, 10, 26, 10};
	private static final int HDF5_FILE_SIGNATURE_LENGTH = HDF5_FILE_SIGNATURE.length;

	public abstract int getVersionOfSuperblock();

	public abstract int getSizeOfOffsets();

	public abstract int getSizeOfLengths();

	public abstract long getBaseAddressByte();

	public abstract long getEndOfFileAddress();

	static boolean verifySignature(FileChannel fc, long offset) {

		// Format Signature
		ByteBuffer signatureBuffer = ByteBuffer.allocate(HDF5_FILE_SIGNATURE_LENGTH);

		try {
			fc.read(signatureBuffer, offset);
		}
		catch(IOException e) {
			throw new HdfException("Failed to read from address: " + Utils.toHex(offset), e);
		}
		// Verify signature
		return Arrays.equals(HDF5_FILE_SIGNATURE, signatureBuffer.array());
	}

	public static Superblock createSuperblock(byte version, long row, long col) {

		switch(version) {
			case 0:
			case 1:
				SuperblockV0V1 sb01 = new SuperblockV0V1();
				sb01.versionOfSuperblock = 0;
				sb01.versionNumberOfTheFileFreeSpaceInformation = 0;
				sb01.versionOfRootGroupSymbolTableEntry = 0;
				sb01.versionOfSharedHeaderMessageFormat = 0;
				sb01.sizeOfOffsets = 8;
				sb01.sizeOfLengths = 8;
				sb01.groupLeafNodeK = 4;
				sb01.groupInternalNodeK = 16;
				sb01.baseAddressByte = 0;
				sb01.addressOfGlobalFreeSpaceIndex = -1;
				sb01.endOfFileAddress = 2048 + (row * col * 8); // long value
				sb01.driverInformationBlockAddress = -1;
				sb01.rootGroupSymbolTableAddress = 56;
				return sb01;

			case 2:
			case 3:
				SuperblockV2V3 sb23 = new SuperblockV2V3();
				return sb23;
			default:
				throw new UnsupportedHdfException("Superblock version is not supported. Detected version = " + version);
		}
	}

	public static Superblock readSuperblock(FileChannel fc, long address) {

		final boolean verifiedSignature = verifySignature(fc, address);
		if(!verifiedSignature) {
			throw new HdfException("Superblock didn't contain valid signature");
		}
		// Signature is ok read rest of Superblock
		long fileLocation = address + HDF5_FILE_SIGNATURE_LENGTH;

		ByteBuffer version = ByteBuffer.allocate(1);
		try {
			fc.read(version, fileLocation);
		}
		catch(IOException e) {
			throw new HdfException("Failed to read superblock at address = " + Utils.toHex(address));
		}
		version.rewind();

		// Version # of Superblock
		final byte versionOfSuperblock = version.get();
		switch(versionOfSuperblock) {

			case 0: // Version 0 is the default format.
			case 1: // Version 1 is the same as version 0 but with the “Indexed Storage Internal Node K”
				// field for storing non-default B-tree ‘K’ value.
				return new SuperblockV0V1(fc, fileLocation);

			case 2: // Version 2 has some fields eliminated and compressed from superblock format versions 0 and 1.
				// It has added checksum support and superblock extension to store additional superblock metadata.
			case 3: // Version 3 is the same as version 2 except that the field “File Consistency Flags” is used
				// for file locking. This format version will enable support for the latest version.
				return new SuperblockV2V3(fc, fileLocation);
			default:
				throw new UnsupportedHdfException(
					"Superblock version is not supported. Detected version = " + versionOfSuperblock);
		}
	}

	public static class SuperblockV0V1 extends Superblock {

		private int versionOfSuperblock;
		private int versionNumberOfTheFileFreeSpaceInformation;
		private int versionOfRootGroupSymbolTableEntry;
		private int versionOfSharedHeaderMessageFormat;
		private int sizeOfOffsets;
		private int sizeOfLengths;
		private int groupLeafNodeK;
		private int groupInternalNodeK;
		private long baseAddressByte;
		private long addressOfGlobalFreeSpaceIndex;
		private long endOfFileAddress;
		private long driverInformationBlockAddress;
		private long rootGroupSymbolTableAddress;

		public SuperblockV0V1() {}

		private SuperblockV0V1(FileChannel fc, long address) {
			try {
				ByteBuffer header = ByteBuffer.allocate(12);
				fc.read(header, address);
				address += 12;

				header.order(LITTLE_ENDIAN);
				header.rewind();

				// Version # of Superblock
				versionOfSuperblock = header.get();

				if(versionOfSuperblock != 0 && versionOfSuperblock != 1) {
					throw new HdfException("Detected superblock version not 0 or 1");
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
				address += 4;

				// Version 1
				if(versionOfSuperblock == 1) {
					// Skip Indexed Storage Internal Node K and zeros
					address += 4;
				}
				int nextSectionSize = 4 * sizeOfOffsets;
				header = ByteBuffer.allocate(nextSectionSize);
				fc.read(header, address);
				address += nextSectionSize;
				header.order(LITTLE_ENDIAN);
				header.rewind();

				// Base Address
				baseAddressByte = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Address of Global Free-space Index
				addressOfGlobalFreeSpaceIndex = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// End of File Address
				endOfFileAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Driver Information Block Address
				driverInformationBlockAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Root Group Symbol Table Entry Address
				rootGroupSymbolTableAddress = address;

			}
			catch(IOException e) {
				throw new HdfException("Failed to read superblock from address " + toHex(address), e);
			}

		}

		@Override public int getVersionOfSuperblock() {
			return versionOfSuperblock;
		}

		public int getVersionNumberOfTheFileFreeSpaceInformation() {
			return versionNumberOfTheFileFreeSpaceInformation;
		}

		public int getVersionOfRootGroupSymbolTableEntry() {
			return versionOfRootGroupSymbolTableEntry;
		}

		public int getVersionOfSharedHeaderMessageFormat() {
			return versionOfSharedHeaderMessageFormat;
		}

		@Override public int getSizeOfOffsets() {
			return sizeOfOffsets;
		}

		@Override public int getSizeOfLengths() {
			return sizeOfLengths;
		}

		public int getGroupLeafNodeK() {
			return groupLeafNodeK;
		}

		public int getGroupInternalNodeK() {
			return groupInternalNodeK;
		}

		@Override public long getBaseAddressByte() {
			return baseAddressByte;
		}

		public long getAddressOfGlobalFreeSpaceIndex() {
			return addressOfGlobalFreeSpaceIndex;
		}

		@Override public long getEndOfFileAddress() {
			return endOfFileAddress;
		}

		public long getDriverInformationBlockAddress() {
			return driverInformationBlockAddress;
		}

		public long getRootGroupSymbolTableAddress() {
			return rootGroupSymbolTableAddress;
		}

		public BufferBuilder toBuffer() {

			BufferBuilder header = new BufferBuilder();
			long address = 12;

			// HDF5 File Signature (8 bytes)
			header.writeBytes(HDF5_FILE_SIGNATURE);

			// Version # of Superblock
			header.writeByte(versionOfSuperblock);

			// Version # of File Free-space Storage
			header.writeByte(versionNumberOfTheFileFreeSpaceInformation);

			// Version # of Root Group Symbol Table Entry
			header.writeByte(versionOfRootGroupSymbolTableEntry);

			// Skip reserved byte
			header.writeByte(0);

			// Version # of Shared Header Message Format
			header.writeByte(versionOfSharedHeaderMessageFormat);

			// Size of Offsets
			header.writeByte(sizeOfOffsets);

			// Size of Lengths
			header.writeByte(sizeOfLengths);

			// Skip reserved byte
			header.writeByte(0);

			// Group Leaf Node K
			header.writeShort((short) groupLeafNodeK);

			// Group Internal Node K
			header.writeShort((short) groupInternalNodeK);

			// File Consistency Flags (skip)
			header.writeInt(0);
			address += 4;

			// Version 1
			if(versionOfSuperblock == 1) {
				// Skip Indexed Storage Internal Node K and zeros
				address += 4;
				header.writeInt(0);
			}

			int nextSectionSize = 4 * sizeOfOffsets;
			address += nextSectionSize;

			// Base Address
			header.writeLong(baseAddressByte);

			// Address of Global Free-space Index
			header.writeLong(addressOfGlobalFreeSpaceIndex);

			// End of File Address
			header.writeLong(endOfFileAddress);

			// Driver Information Block Address
			header.writeLong(driverInformationBlockAddress);

			return header;
		}
	}

	public static class SuperblockV2V3 extends Superblock {

		private int versionOfSuperblock;
		private int sizeOfOffsets;
		private int sizeOfLengths;
		private long baseAddressByte;
		private long superblockExtensionAddress;
		private long endOfFileAddress;
		private long rootGroupObjectHeaderAddress;

		public SuperblockV2V3() {
		}

		private SuperblockV2V3(FileChannel fc, final long address) {
			try {

				ByteBuffer header = ByteBuffer.allocate(4);
				fc.read(header, address);
				header.order(LITTLE_ENDIAN);
				header.rewind();

				// Version # of Superblock
				versionOfSuperblock = header.get();

				// Size of Offsets
				sizeOfOffsets = Byte.toUnsignedInt(header.get());

				// Size of Lengths
				sizeOfLengths = Byte.toUnsignedInt(header.get());

				// TODO File consistency flags

				int nextSectionSize = 4 * sizeOfOffsets + 4;
				header = ByteBuffer.allocate(nextSectionSize);
				fc.read(header, address + 4);
				header.order(LITTLE_ENDIAN);
				header.rewind();

				// Base Address
				baseAddressByte = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Superblock Extension Address
				superblockExtensionAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				if(superblockExtensionAddress != Constants.UNDEFINED_ADDRESS) {
					throw new UnsupportedHdfException("Superblock extension is not supported");
				}

				// End of File Address
				endOfFileAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Root Group Object Header Address
				rootGroupObjectHeaderAddress = Utils.readBytesAsUnsignedLong(header, sizeOfOffsets);

				// Validate checksum
				// 12 = 8 bytes for signature + 4 for first part of superblock
				ByteBuffer superblockBuffer = ByteBuffer.allocate(nextSectionSize + 12);
				fc.read(superblockBuffer, address - 8); // -8 for signature
				superblockBuffer.order(LITTLE_ENDIAN);
				superblockBuffer.rewind();
				ChecksumUtils.validateChecksum(superblockBuffer);

			}
			catch(IOException e) {
				throw new HdfException("Failed to read superblock from address " + toHex(address), e);
			}
		}

		public SuperblockV2V3(long baseAddressByte, long rootGroupObjectHeaderAddress) {
			this.versionOfSuperblock = 3;
			this.sizeOfOffsets = 8;
			this.sizeOfLengths = 8;
			this.baseAddressByte = baseAddressByte;
			this.rootGroupObjectHeaderAddress = rootGroupObjectHeaderAddress;
			this.superblockExtensionAddress = Constants.UNDEFINED_ADDRESS;
			this.endOfFileAddress = Constants.UNDEFINED_ADDRESS;
		}

		@Override public int getVersionOfSuperblock() {
			return versionOfSuperblock;
		}

		@Override public int getSizeOfOffsets() {
			return sizeOfOffsets;
		}

		@Override public int getSizeOfLengths() {
			return sizeOfLengths;
		}

		@Override public long getBaseAddressByte() {
			return baseAddressByte;
		}

		public long getSuperblockExtensionAddress() {
			return superblockExtensionAddress;
		}

		@Override public long getEndOfFileAddress() {
			return endOfFileAddress;
		}

		public long getRootGroupObjectHeaderAddress() {
			return rootGroupObjectHeaderAddress;
		}

		public BufferBuilder toBuffer() {

			BufferBuilder bufferBuilder = new BufferBuilder().writeBytes(HDF5_FILE_SIGNATURE)
				.writeByte(versionOfSuperblock).writeByte(sizeOfOffsets).writeByte(sizeOfLengths)
				.writeByte(0) // file consistency flags
				.writeLong(baseAddressByte).writeLong(superblockExtensionAddress).writeLong(endOfFileAddress)
				.writeLong(rootGroupObjectHeaderAddress).appendChecksum();

			return bufferBuilder;
		}
	}
}
