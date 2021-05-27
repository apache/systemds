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


package org.apache.sysds.runtime.io.hdf5.btree;

import org.apache.sysds.runtime.io.hdf5.BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

public abstract class BTreeV1 {

	private static final byte[] BTREE_NODE_V1_SIGNATURE = "TREE".getBytes(StandardCharsets.US_ASCII);
	private static final int HEADER_BYTES = 6;

	private final long address;
	protected final int entriesUsed;
	private final long leftSiblingAddress;
	private final long rightSiblingAddress;
	private byte nodeType;
	private byte nodeLevel;
	protected int sizeOfLengths;
	protected int sizeOfOffsets;

	public static BTreeV1Group createGroupBTree(HdfFileChannel hdfFc, long address) {

		ByteBuffer header = readHeaderAndValidateSignature(hdfFc, address);

		final byte nodeType = header.get();
		if(nodeType != 0) {
			throw new HdfException("B tree type is not group. Type is: " + nodeType);
		}

		final byte nodeLevel = header.get();

		if(nodeLevel > 0) {
			return new BTreeV1Group.BTreeV1GroupNonLeafNode(hdfFc, address);
		}
		else {
			return new BTreeV1Group.BTreeV1GroupLeafNode(hdfFc, address);
		}

	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}

	public BufferBuilder toBuffer(BufferBuilder header) {
		writeHeaderSignature(header);

		if(nodeType != 0) {
			throw new HdfException("B tree type is not group. Type is: " + nodeType);
		}

		header.writeByte(nodeType);
		header.writeByte(nodeLevel);

		header.writeShort((short) entriesUsed);
		header.writeLong(leftSiblingAddress);
		header.writeLong(rightSiblingAddress);
		return header;

	}

	public static BTreeV1Data createDataBTree(HdfFileChannel hdfFc, long address, int dataDimensions) {
		ByteBuffer header = readHeaderAndValidateSignature(hdfFc, address);

		final byte nodeType = header.get();
		if(nodeType != 1) {
			throw new HdfException("B tree type is not data. Type is: " + nodeType);
		}

		final byte nodeLevel = header.get();

		if(nodeLevel > 0) {
			return new BTreeV1Data.BTreeV1DataNonLeafNode(hdfFc, address, dataDimensions);
		}
		else {
			return new BTreeV1Data.BTreeV1DataLeafNode(hdfFc, address, dataDimensions);
		}
	}

	public static ByteBuffer readHeaderAndValidateSignature(HdfFileChannel fc, long address) {
		ByteBuffer header = fc.readBufferFromAddress(address, HEADER_BYTES);

		// Verify signature
		byte[] formatSignatureByte = new byte[4];
		header.get(formatSignatureByte, 0, formatSignatureByte.length);
		if(!Arrays.equals(BTREE_NODE_V1_SIGNATURE, formatSignatureByte)) {
			throw new HdfException("B tree V1 node signature not matched");
		}
		return header;
	}

	private static BufferBuilder writeHeaderSignature(BufferBuilder header) {
		header.writeBytes(BTREE_NODE_V1_SIGNATURE);
		return header;
	}

	public BTreeV1(HdfFileChannel hdfFc, long address) {
		this.address = address;

		int headerSize = 8 * hdfFc.getSizeOfOffsets();
		ByteBuffer header = hdfFc.readBufferFromAddress(address + 6, headerSize);

		entriesUsed = Utils.readBytesAsUnsignedInt(header, 2);

		leftSiblingAddress = Utils.readBytesAsUnsignedLong(header, hdfFc.getSizeOfOffsets());

		rightSiblingAddress = Utils.readBytesAsUnsignedLong(header, hdfFc.getSizeOfOffsets());

	}

	public BTreeV1(byte nodeType, byte nodeLevel, int entriesUsed, long leftSiblingAddress, long rightSiblingAddress,
		int sizeOfLengths, int sizeOfOffsets) {
		this.address = 0;
		this.nodeType = nodeType;
		this.nodeLevel = nodeLevel;
		this.entriesUsed = entriesUsed;
		this.leftSiblingAddress = leftSiblingAddress;
		this.rightSiblingAddress = rightSiblingAddress;
		this.sizeOfLengths = sizeOfLengths;
		this.sizeOfOffsets = sizeOfOffsets;
	}

	public int getEntriesUsed() {
		return entriesUsed;
	}

	public long getLeftSiblingAddress() {
		return leftSiblingAddress;
	}

	public long getRightSiblingAddress() {
		return rightSiblingAddress;
	}

	public long getAddress() {
		return address;
	}

	public abstract List<Long> getChildAddresses();



}
