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

import static org.apache.sysds.runtime.io.hdf5.Utils.toHex;

public class SymbolTableEntry {
	/**
	 * The location of this symbol table entry in the file
	 */
	private final long address;
	private final int linkNameOffset;
	private final long objectHeaderAddress;
	private final int cacheType;
	private long bTreeAddress = -1;
	private long nameHeapAddress = -1;
	private int linkValueOffset = -1;

	public SymbolTableEntry(HdfFileChannel fc) {

		this.address = fc.getRootGroupAddress();
		this.linkNameOffset = 0;
		this.objectHeaderAddress = 96;
		this.cacheType = 1;

		//"linkNameOffset":0,"objectHeaderAddress":96,"cacheType":1,"bTreeAddress":136,"nameHeapAddress":680,"linkValueOffset":-1}
		switch(cacheType) {
			case 0:
				// Nothing in scratch pad space
				break;
			case 1:
				// B Tree
				// Address of B Tree
				this.bTreeAddress = 136;
				this.nameHeapAddress = 680;
				break;
			case 2:
				// Link
				//this.linkValueOffset = Utils.readBytesAsUnsignedInt(bb, 4);
				// TODO: find out the values
				break;
			default:
				throw new IllegalStateException("SymbolTableEntry: Unrecognized cache type = " + cacheType);
		}
	}

	public SymbolTableEntry(int linkNameOffset, long objectHeaderAddress, int cacheType, long bTreeAddress,
		long nameHeapAddress, int linkValueOffset) {

		this.address = 0;
		this.linkNameOffset = linkNameOffset;
		this.objectHeaderAddress = objectHeaderAddress;
		this.cacheType = cacheType;
		this.bTreeAddress = bTreeAddress;
		this.nameHeapAddress = nameHeapAddress;
		this.linkValueOffset = linkValueOffset;
	}

	public BufferBuilder toBuffer() {
		BufferBuilder bufferBuilder = new BufferBuilder();
		return this.toBuffer(bufferBuilder);
	}

	public BufferBuilder toBuffer(BufferBuilder bufferBuilder) {
		bufferBuilder.writeLong(this.linkNameOffset);
		bufferBuilder.writeLong(this.objectHeaderAddress);
		bufferBuilder.writeInt(this.cacheType);

		// Reserved 4 bytes
		bufferBuilder.writeInt(0);

		// Scratch pad
		switch(this.cacheType) {
			case 0:
				// Nothing in scratch pad space
				break;
			case 1:
				// B Tree
				// Address of B Tree
				bufferBuilder.writeLong(this.bTreeAddress);

				// Address of Name Heap
				bufferBuilder.writeLong(this.nameHeapAddress);
				break;
			case 2:
				// Link
				bufferBuilder.writeInt(this.linkValueOffset);
				break;
			default:
				throw new IllegalStateException("SymbolTableEntry: Unrecognized cache type = " + cacheType);
		}
		return bufferBuilder;
	}

	public SymbolTableEntry(HdfFileChannel fc, long address) {
		this.address = address;

		final int size = fc.getSizeOfOffsets() * 2 + 4 + 4 + 16;

		final ByteBuffer bb = fc.readBufferFromAddress(address, size);

		// Link Name Offset
		linkNameOffset = Utils.readBytesAsUnsignedInt(bb, fc.getSizeOfOffsets());

		// Object Header Address
		objectHeaderAddress = Utils.readBytesAsUnsignedLong(bb, fc.getSizeOfOffsets());

		// Link Name Offset
		cacheType = Utils.readBytesAsUnsignedInt(bb, 4);

		// Reserved 4 bytes
		bb.get(new byte[4]);

		// Scratch pad
		switch(cacheType) {
			case 0:
				// Nothing in scratch pad space
				break;
			case 1:
				// B Tree
				// Address of B Tree
				bTreeAddress = Utils.readBytesAsUnsignedLong(bb, fc.getSizeOfOffsets());

				// Address of Name Heap
				nameHeapAddress = Utils.readBytesAsUnsignedLong(bb, fc.getSizeOfOffsets());
				break;
			case 2:
				// Link
				linkValueOffset = Utils.readBytesAsUnsignedInt(bb, 4);
				break;
			default:
				throw new IllegalStateException("SymbolTableEntry: Unrecognized cache type = " + cacheType);
		}

	}

	public long getAddress() {
		return address;
	}

	public long getBTreeAddress() {
		return bTreeAddress;
	}

	public int getCacheType() {
		return cacheType;
	}

	public int getLinkNameOffset() {
		return linkNameOffset;
	}

	public int getLinkValueOffset() {
		return linkValueOffset;
	}

	public long getNameHeapAddress() {
		return nameHeapAddress;
	}

	public long getObjectHeaderAddress() {
		return objectHeaderAddress;
	}

	@Override public String toString() {
		return "SymbolTableEntry [address=" + toHex(
			address) + ", linkNameOffset=" + linkNameOffset + ", objectHeaderAddress=" + toHex(
			objectHeaderAddress) + ", cacheType=" + cacheType + ", bTreeAddress=" + toHex(
			bTreeAddress) + ", nameHeapAddress=" + toHex(
			nameHeapAddress) + ", linkValueOffset=" + linkValueOffset + "]";
	}

}
