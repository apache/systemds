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

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5GroupSymbolTableNode {

	private static final byte[] NODE_SIGNATURE = "SNOD".getBytes(StandardCharsets.US_ASCII);

	private final short numberOfEntries;
	private final H5SymbolTableEntry[] symbolTableEntries;
	private final int headerSize = 8;
	private final H5RootObject rootObject;

	public H5GroupSymbolTableNode(H5RootObject rootObject, long address) {
		this.rootObject = rootObject;
		try {

			ByteBuffer header = rootObject.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			header.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if(!Arrays.equals(NODE_SIGNATURE, formatSignatureBytes)) {
				throw new H5RuntimeException("Group symbol table Node signature not matched");
			}

			// Version Number
			rootObject.setGroupSymbolTableNodeVersion(header.get());

			// Move past reserved space
			header.position(6);

			final byte[] twoBytes = new byte[2];

			// Data Segment Size
			header.get(twoBytes);
			numberOfEntries = ByteBuffer.wrap(twoBytes).order(LITTLE_ENDIAN).getShort();

			final long symbolTableEntryBytes = rootObject.getSuperblock().sizeOfOffsets * 2L + 8L + 16L;

			symbolTableEntries = new H5SymbolTableEntry[numberOfEntries];
			for(int i = 0; i < numberOfEntries; i++) {
				long offset = address + headerSize + i * symbolTableEntryBytes;
				symbolTableEntries[i] = new H5SymbolTableEntry(rootObject, offset);
			}
		}
		catch(Exception e) {
			throw new H5RuntimeException("Error reading Group symbol table node", e);
		}
	}

	public H5GroupSymbolTableNode(H5RootObject rootObject, short numberOfEntries,
		H5SymbolTableEntry[] symbolTableEntries) {
		this.rootObject = rootObject;
		this.numberOfEntries = numberOfEntries;
		this.symbolTableEntries = symbolTableEntries;
	}

	public void toBuffer(H5BufferBuilder bb) {

		bb.writeBytes(NODE_SIGNATURE);
		bb.writeByte(rootObject.getGroupSymbolTableNodeVersion());

		// Move past reserved space
		bb.writeByte(0);

		bb.writeShort(numberOfEntries);

		for(H5SymbolTableEntry symbolTableEntry : symbolTableEntries) {
			symbolTableEntry.toBuffer(bb);
		}

	}

	public short getVersion() {
		return rootObject.getGroupSymbolTableNodeVersion();
	}

	public short getNumberOfEntries() {
		return numberOfEntries;
	}

	public H5SymbolTableEntry[] getSymbolTableEntries() {
		return ArrayUtils.clone(symbolTableEntries);
	}
}
