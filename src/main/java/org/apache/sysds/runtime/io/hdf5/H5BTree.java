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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class H5BTree {

	private static final byte[] BTREE_NODE_SIGNATURE = "TREE".getBytes(StandardCharsets.US_ASCII);
	private static final int HEADER_BYTES = 6;
	@SuppressWarnings("unused")
	private final long address;
	protected final int entriesUsed;
	private final long leftSiblingAddress;
	private final long rightSiblingAddress;
	private byte nodeType;
	private byte nodeLevel;
	private final List<Long> childAddresses;
	private final H5RootObject rootObject;

	public H5BTree(H5RootObject rootObject, byte nodeType, byte nodeLevel, int entriesUsed, long leftSiblingAddress,
		long rightSiblingAddress, List<Long> childAddresses) {
		this.address = 0;
		this.rootObject = rootObject;
		this.nodeType = nodeType;
		this.nodeLevel = nodeLevel;
		this.entriesUsed = entriesUsed;
		this.leftSiblingAddress = leftSiblingAddress;
		this.rightSiblingAddress = rightSiblingAddress;
		this.childAddresses = childAddresses;
	}

	public H5BTree(H5RootObject rootObject, long address) {

		this.address = address;
		this.rootObject = rootObject;

		readHeaderAndValidateSignature(rootObject, address);

		int headerSize = 8 * rootObject.getSuperblock().sizeOfOffsets;
		ByteBuffer header = rootObject.readBufferFromAddress(address + 6, headerSize);

		this.entriesUsed = Utils.readBytesAsUnsignedInt(header, 2);

		this.leftSiblingAddress = Utils.readBytesAsUnsignedLong(header, rootObject.getSuperblock().sizeOfOffsets);

		this.rightSiblingAddress = Utils.readBytesAsUnsignedLong(header, rootObject.getSuperblock().sizeOfOffsets);

		final int keyBytes = (2 * entriesUsed + 1) * rootObject.getSuperblock().sizeOfLengths;
		final int childPointerBytes = (2 * entriesUsed) * rootObject.getSuperblock().sizeOfLengths;
		final int keysAndPointersBytes = keyBytes + childPointerBytes;

		final long keysAddress = address + 8L + 2L * rootObject.getSuperblock().sizeOfOffsets;
		final ByteBuffer keysAndPointersBuffer = rootObject.readBufferFromAddress(keysAddress, keysAndPointersBytes);

		childAddresses = new ArrayList<>(entriesUsed);

		for(int i = 0; i < entriesUsed; i++) {
			keysAndPointersBuffer.position(keysAndPointersBuffer.position() + rootObject.getSuperblock().sizeOfLengths);
			childAddresses
				.add(Utils.readBytesAsUnsignedLong(keysAndPointersBuffer, rootObject.getSuperblock().sizeOfLengths));
		}

	}

	public H5BufferBuilder toBuffer() {
		H5BufferBuilder header = new H5BufferBuilder();
		toBuffer(header);
		return header;
	}

	public void toBuffer(H5BufferBuilder bb) {

		bb.writeBytes(BTREE_NODE_SIGNATURE);

		if(nodeType != 0) {
			throw new H5RuntimeException("B tree type is not group. Type is: " + nodeType);
		}

		bb.writeByte(nodeType);
		bb.writeByte(nodeLevel);
		bb.writeShort((short) entriesUsed);
		bb.writeLong(leftSiblingAddress);
		bb.writeLong(rightSiblingAddress);

		for(int i = 0; i < entriesUsed; i++) {
			bb.writeBytes(new byte[rootObject.getSuperblock().sizeOfLengths]);
			bb.writeLong(this.childAddresses.get(i));
		}

	}

	public static ByteBuffer readHeaderAndValidateSignature(H5RootObject rootObject, long address) {
		ByteBuffer header = rootObject.readBufferFromAddress(address, HEADER_BYTES);

		// Verify signature
		byte[] formatSignatureByte = new byte[4];
		header.get(formatSignatureByte, 0, formatSignatureByte.length);
		if(!Arrays.equals(BTREE_NODE_SIGNATURE, formatSignatureByte)) {
			throw new H5RuntimeException("B tree node signature not matched");
		}
		return header;
	}

	public List<Long> getChildAddresses() {
		return childAddresses;
	}
}
