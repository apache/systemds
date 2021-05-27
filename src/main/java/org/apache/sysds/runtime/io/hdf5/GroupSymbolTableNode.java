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

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.commons.lang3.ArrayUtils;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class GroupSymbolTableNode {

	private static final byte[] NODE_SIGNATURE = "SNOD".getBytes(StandardCharsets.US_ASCII);

	/** The location of this GroupSymbolTableNode in the file */
	private final long address;
	private final short version;
	private final short numberOfEntries;
	private final SymbolTableEntry[] symbolTableEntries;
	private final int headerSize = 8;

	public GroupSymbolTableNode(HdfFileChannel hdfFc, long address) {
		this.address = address;
		try {

			ByteBuffer header = hdfFc.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			header.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if (!Arrays.equals(NODE_SIGNATURE, formatSignatureBytes)) {
				throw new HdfException("Group symbol table Node signature not matched");
			}

			// Version Number
			version = header.get();

			// Move past reserved space
			header.position(6);

			final byte[] twoBytes = new byte[2];

			// Data Segment Size
			header.get(twoBytes);
			numberOfEntries = ByteBuffer.wrap(twoBytes).order(LITTLE_ENDIAN).getShort();

			final long symbolTableEntryBytes = hdfFc.getSizeOfOffsets() * 2L + 8L + 16L;

			symbolTableEntries = new SymbolTableEntry[numberOfEntries];
			for (int i = 0; i < numberOfEntries; i++) {
				long offset = address + headerSize + i * symbolTableEntryBytes;
				symbolTableEntries[i] = new SymbolTableEntry(hdfFc, offset);
			}
		} catch (Exception e) {
			// TODO improve message
			throw new HdfException("Error reading Group symbol table node", e);
		}
	}
	public GroupSymbolTableNode(byte version, short numberOfEntries, SymbolTableEntry[] symbolTableEntries) {
		this.address = 0;
		this.version = version;
		this. numberOfEntries = numberOfEntries;
		this.symbolTableEntries = symbolTableEntries;
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}
	public BufferBuilder toBuffer(BufferBuilder header) {

		header.writeBytes(NODE_SIGNATURE);
		header.writeByte(version);

		// Move past reserved space
		header.writeByte(0);

		header.writeShort(numberOfEntries);

		for(SymbolTableEntry symbolTableEntry : symbolTableEntries) {
			symbolTableEntry.toBuffer(header);
		}
		return header;
	}

	public short getVersion() {
		return version;
	}

	public short getNumberOfEntries() {
		return numberOfEntries;
	}

	public SymbolTableEntry[] getSymbolTableEntries() {
		return ArrayUtils.clone(symbolTableEntries);
	}

	@Override
	public String toString() {
		return "GroupSymbolTableNode [address=" + address + ", numberOfEntries=" + numberOfEntries + "]";
	}

}
