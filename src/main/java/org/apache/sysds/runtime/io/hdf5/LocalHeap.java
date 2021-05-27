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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class LocalHeap {

	private static final byte[] HEAP_SIGNATURE = "HEAP".getBytes(StandardCharsets.US_ASCII);

	/** The location of this Heap in the file */
	private final long address;
	private final short version;
	private final long dataSegmentSize;
	private final long offsetToHeadOfFreeList;
	private final long addressOfDataSegment;
	private final ByteBuffer dataBuffer;

	public LocalHeap(HdfFileChannel hdfFc, long address) {
		this.address = address;
		try {
			// Header
			int headerSize = 8 + hdfFc.getSizeOfLengths() + hdfFc.getSizeOfLengths() + hdfFc.getSizeOfOffsets();
			ByteBuffer header = hdfFc.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			header.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if (!Arrays.equals(HEAP_SIGNATURE, formatSignatureBytes)) {
				throw new HdfException("Heap signature not matched");
			}

			// Version
			version = header.get();

			// Move past reserved space
			header.position(8);

			// Data Segment Size
			dataSegmentSize = Utils.readBytesAsUnsignedLong(header, hdfFc.getSizeOfLengths());

			// Offset to Head of Free-list
			offsetToHeadOfFreeList = Utils.readBytesAsUnsignedLong(header, hdfFc.getSizeOfLengths());

			// Address of Data Segment
			addressOfDataSegment = Utils.readBytesAsUnsignedLong(header, hdfFc.getSizeOfOffsets());

			dataBuffer = hdfFc.map(addressOfDataSegment, dataSegmentSize);

		} catch (Exception e) {
			throw new HdfException("Error reading local heap", e);
		}
	}

	public  LocalHeap(short version, String childName,long dataSegmentSize, long offsetToHeadOfFreeList, long addressOfDataSegment){
		this.address = 0;
		this.version = version;
		this.dataSegmentSize = dataSegmentSize;
		this.offsetToHeadOfFreeList = offsetToHeadOfFreeList;
		this.addressOfDataSegment = addressOfDataSegment;

		byte[] childName_atBytes = childName.getBytes(Charset.forName("UTF-8"));

		byte[] nullValues= new byte[(int) (dataSegmentSize- childName_atBytes.length)];
	    Arrays.fill(nullValues, Constants.NULL);
		this.dataBuffer = ByteBuffer.allocate((int) this.dataSegmentSize);
		this.dataBuffer.put(childName_atBytes);
		this.dataBuffer.put(nullValues);
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}

	public BufferBuilder toBuffer(BufferBuilder header) {
		header.writeBytes(HEAP_SIGNATURE);

		header.writeByte((byte)this.version);

		// Move past reserved space
		header.writeBytes(new byte[3]);

		header.writeLong(dataSegmentSize);

		header.writeLong(offsetToHeadOfFreeList);

		header.writeLong(addressOfDataSegment);

		header.writeBytes(dataBuffer.array());

		return header;
	}

	public short getVersion() {
		return version;
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

	@Override
	public String toString() {
		return "LocalHeap [address=" + Utils.toHex(address) + ", version=" + version + ", dataSegmentSize="
				+ dataSegmentSize + ", offsetToHeadOfFreeList=" + offsetToHeadOfFreeList + ", addressOfDataSegment="
				+ Utils.toHex(addressOfDataSegment) + "]";
	}

	public ByteBuffer getDataBuffer() {
		return dataBuffer;
	}

}
