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
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class H5LocalHeap {

	private static final byte[] HEAP_SIGNATURE = "HEAP".getBytes(StandardCharsets.US_ASCII);
	private final long dataSegmentSize;
	private final long offsetToHeadOfFreeList;
	private final long addressOfDataSegment;
	private final ByteBuffer dataBuffer;
	private final H5RootObject rootObject;

	public H5LocalHeap(H5RootObject rootObject, long address) {
		try {
			this.rootObject = rootObject;
			int sizeOfLength = rootObject.getSuperblock().sizeOfLengths;
			int sizeOfOffset = rootObject.getSuperblock().sizeOfOffsets;
			// Header
			int headerSize = 8 + sizeOfLength + sizeOfLength + sizeOfOffset;
			ByteBuffer header = rootObject.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			header.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if(!Arrays.equals(HEAP_SIGNATURE, formatSignatureBytes)) {
				throw new H5RuntimeException("Heap signature not matched");
			}

			// Version
			rootObject.setLocalHeapVersion(header.get());

			// Move past reserved space
			header.position(8);

			// Data Segment Size
			dataSegmentSize = Utils.readBytesAsUnsignedLong(header, sizeOfLength);

			// Offset to Head of Free-list
			offsetToHeadOfFreeList = Utils.readBytesAsUnsignedLong(header, sizeOfLength);

			// Address of Data Segment
			addressOfDataSegment = Utils.readBytesAsUnsignedLong(header, sizeOfOffset);

			dataBuffer = rootObject.readBufferFromAddress(addressOfDataSegment, (int) dataSegmentSize);

		}
		catch(Exception e) {
			throw new H5RuntimeException("Error reading local heap", e);
		}
	}

	public H5LocalHeap(H5RootObject rootObject, String childName, long dataSegmentSize, long offsetToHeadOfFreeList,
		long addressOfDataSegment) {
		this.rootObject = rootObject;
		this.dataSegmentSize = dataSegmentSize;

		int blockCount = (int) (childName.length() / 8f + 1) + 1;
		this.offsetToHeadOfFreeList = blockCount * 8L; //offsetToHeadOfFreeList;
		this.addressOfDataSegment = addressOfDataSegment;
		byte[] childName_atBytes = childName.getBytes(StandardCharsets.US_ASCII);
		this.dataBuffer = ByteBuffer.allocate((int) this.dataSegmentSize);
		this.dataBuffer.position(8);
		this.dataBuffer.put(childName_atBytes);
		this.dataBuffer.put(H5Constants.NULL);

		this.dataBuffer.position(blockCount * 8 - 1);
		this.dataBuffer.putShort((short) 1);
		this.dataBuffer.position((int) (this.offsetToHeadOfFreeList + 8 - 1));
		this.dataBuffer.putShort((short) (this.dataSegmentSize - this.offsetToHeadOfFreeList));
	}

	public void toBuffer(H5BufferBuilder bb) {

		bb.writeBytes(HEAP_SIGNATURE);

		bb.writeByte(rootObject.localHeapVersion);

		// Move past reserved space
		bb.writeBytes(new byte[3]);

		bb.writeLong(dataSegmentSize);

		bb.writeLong(offsetToHeadOfFreeList);

		bb.writeLong(addressOfDataSegment);

		bb.writeBytes(dataBuffer.array());

	}

	public short getVersion() {
		return rootObject.localHeapVersion;
	}

	public long getDataSegmentSize() {
		return dataSegmentSize;
	}

	public long getOffsetToHeadOfFreeList() {
		return offsetToHeadOfFreeList;
	}

	public long getAddressOfDataSegment() {
		return addressOfDataSegment;
	}

	public ByteBuffer getDataBuffer() {
		return dataBuffer;
	}

}
