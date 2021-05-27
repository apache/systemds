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

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public abstract class BTreeV1Group extends BTreeV1 {

	private BTreeV1Group(HdfFileChannel hdfFc, long address) {
		super(hdfFc, address);
	}

	public BTreeV1Group(byte nodeType, byte nodeLevel, int entriesUsed, long leftSiblingAddress,
		long rightSiblingAddress, int sizeOfLengths, int sizeOfOffsets) {
		super(nodeType, nodeLevel, entriesUsed, leftSiblingAddress, rightSiblingAddress, sizeOfLengths, sizeOfOffsets);
	}

	public static class BTreeV1GroupLeafNode extends BTreeV1Group {

		private final List<Long> childAddresses;


		public BTreeV1GroupLeafNode(HdfFileChannel hdfFc, long address) {
			super(hdfFc, address);

			final int keyBytes = (2 * entriesUsed + 1) * hdfFc.getSizeOfLengths();
			final int childPointerBytes = (2 * entriesUsed) * hdfFc.getSizeOfOffsets();
			final int keysAndPointersBytes = keyBytes + childPointerBytes;

			final long keysAddress = address + 8L + 2L * hdfFc.getSizeOfOffsets();
			final ByteBuffer keysAndPointersBuffer = hdfFc.readBufferFromAddress(keysAddress, keysAndPointersBytes);

			childAddresses = new ArrayList<>(entriesUsed);

			for(int i = 0; i < entriesUsed; i++) {
				keysAndPointersBuffer.position(keysAndPointersBuffer.position() + hdfFc.getSizeOfLengths());
				childAddresses.add(Utils.readBytesAsUnsignedLong(keysAndPointersBuffer, hdfFc.getSizeOfOffsets()));
			}
		}

		public BTreeV1GroupLeafNode(byte nodeType, byte nodeLevel, int entriesUsed, long leftSiblingAddress,
			long rightSiblingAddress, int sizeOfLengths, int sizeOfOffsets, List<Long> childAddresses) {
			super(nodeType, nodeLevel, entriesUsed, leftSiblingAddress, rightSiblingAddress, sizeOfLengths, sizeOfOffsets);
			this.childAddresses = childAddresses;
		}

		@Override public BufferBuilder toBuffer() {
			BufferBuilder header = new BufferBuilder();
			return toBuffer(header);
		}

		@Override public BufferBuilder toBuffer(BufferBuilder header) {
			super.toBuffer(header);
			for(int i = 0; i < entriesUsed; i++) {
				header.writeBytes(new byte[sizeOfLengths]);
				header.writeLong(this.childAddresses.get(i));
			}
			return header;
		}

		@Override public List<Long> getChildAddresses() {
			return childAddresses;
		}

	}

	static class BTreeV1GroupNonLeafNode extends BTreeV1Group {

		private final List<BTreeV1> childNodes;

		BTreeV1GroupNonLeafNode(HdfFileChannel hdfFc, long address) {
			super(hdfFc, address);

			final int keyBytes = (2 * entriesUsed + 1) * hdfFc.getSizeOfLengths();
			final int childPointerBytes = (2 * entriesUsed) * hdfFc.getSizeOfOffsets();
			final int keysAndPointersBytes = keyBytes + childPointerBytes;

			final long keysAddress = address + 8L + 2L * hdfFc.getSizeOfOffsets();
			final ByteBuffer keysAndPointersBuffer = hdfFc.readBufferFromAddress(keysAddress, keysAndPointersBytes);

			childNodes = new ArrayList<>(entriesUsed);

			for(int i = 0; i < entriesUsed; i++) {
				keysAndPointersBuffer.position(keysAndPointersBuffer.position() + hdfFc.getSizeOfOffsets());
				long childAddress = Utils.readBytesAsUnsignedLong(keysAndPointersBuffer, hdfFc.getSizeOfOffsets());
				childNodes.add(createGroupBTree(hdfFc, childAddress));
			}

		}

		@Override public List<Long> getChildAddresses() {
			List<Long> childAddresses = new ArrayList<>();
			for(BTreeV1 child : childNodes) {
				childAddresses.addAll(child.getChildAddresses());
			}
			return childAddresses;
		}
	}
}
