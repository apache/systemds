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

public class H5SymbolTableEntry {

	private final long address;
	private final int linkNameOffset;
	private final long objectHeaderAddress;
	private final int cacheType;
	private final long bTreeAddress;
	private final long nameHeapAddress;

	public H5SymbolTableEntry(H5RootObject rootObject) {
		this.address = rootObject.getSuperblock().rootGroupSymbolTableAddress;
		this.linkNameOffset = 0;
		this.objectHeaderAddress = 96;
		this.cacheType = 1;
		this.bTreeAddress = 136;
		this.nameHeapAddress = 680;
	}

	public H5SymbolTableEntry(int linkNameOffset, long objectHeaderAddress, int cacheType, long bTreeAddress,
		long nameHeapAddress) {

		this.address = 0;
		this.linkNameOffset = linkNameOffset;
		this.objectHeaderAddress = objectHeaderAddress;
		this.cacheType = cacheType;
		this.bTreeAddress = bTreeAddress;
		this.nameHeapAddress = nameHeapAddress;
	}

	public void toBuffer(H5BufferBuilder bb) {
		bb.writeLong(this.linkNameOffset);
		bb.writeLong(this.objectHeaderAddress);
		bb.writeInt(this.cacheType);

		// Reserved 4 bytes
		bb.writeInt(0);

		if(cacheType == 1) {
			// B Tree
			// Address of B Tree
			bb.writeLong(this.bTreeAddress);

			// Address of Name Heap
			bb.writeLong(this.nameHeapAddress);
		}
	}

	public H5SymbolTableEntry(H5RootObject rootObject, long address) {
		this.address = address;

		final int size = rootObject.getSuperblock().sizeOfOffsets * 2 + 4 + 4 + 16;

		final ByteBuffer bb = rootObject.readBufferFromAddress(address, size);

		// Link Name Offset
		linkNameOffset = Utils.readBytesAsUnsignedInt(bb, rootObject.getSuperblock().sizeOfOffsets);

		// Object Header Address
		objectHeaderAddress = Utils.readBytesAsUnsignedLong(bb, rootObject.getSuperblock().sizeOfOffsets);

		// Link Name Offset
		cacheType = Utils.readBytesAsUnsignedInt(bb, 4);

		// Reserved 4 bytes
		bb.get(new byte[4]);

		// B Tree
		// Address of B Tree
		bTreeAddress = Utils.readBytesAsUnsignedLong(bb, rootObject.getSuperblock().sizeOfOffsets);

		// Address of Name Heap
		nameHeapAddress = Utils.readBytesAsUnsignedLong(bb, rootObject.getSuperblock().sizeOfOffsets);

	}

	public long getAddress() {
		return address;
	}

	public int getLinkNameOffset() {
		return linkNameOffset;
	}

	public long getObjectHeaderAddress() {
		return objectHeaderAddress;
	}

	public int getCacheType() {
		return cacheType;
	}

	public long getbTreeAddress() {
		return bTreeAddress;
	}

	public long getNameHeapAddress() {
		return nameHeapAddress;
	}
}
