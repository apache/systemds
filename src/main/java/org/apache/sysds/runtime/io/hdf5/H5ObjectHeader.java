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

import org.apache.sysds.runtime.io.hdf5.message.H5Message;
import org.apache.sysds.runtime.io.hdf5.message.H5DataTypeMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataSpaceMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5FillValueMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5NilMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5ObjectModificationTimeMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5SymbolTableMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataLayoutMessage;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.io.hdf5.Utils.readBytesAsUnsignedInt;

public class H5ObjectHeader {

	private final H5RootObject rootObject;
	private final List<H5Message> messages = new ArrayList<>();
	private int referenceCount = 1;
	private String datasetName;

	public H5ObjectHeader(H5RootObject rootObject, long address) {
		this.rootObject = rootObject;
		try {
			ByteBuffer header = rootObject.readBufferFromAddress(address, 12);

			// Version
			rootObject.setObjectHeaderVersion(header.get());
			if(rootObject.getObjectHeaderVersion() != 1) {
				throw new H5RuntimeException("Invalid version detected. Version is = " + rootObject.getObjectHeaderVersion());
			}

			// Skip reserved byte
			header.position(header.position() + 1);

			// Number of messages
			final int numberOfMessages = readBytesAsUnsignedInt(header, 2);

			// Reference Count
			referenceCount = readBytesAsUnsignedInt(header, 4);

			// Size of the messages
			int headerSize = readBytesAsUnsignedInt(header, 4);

			// 12 up to this point + 4 missed in format spec = 16
			address += 16;
			header = rootObject.readBufferFromAddress(address, headerSize);

			readMessages(header, numberOfMessages);

		}
		catch(Exception e) {
			throw new H5RuntimeException("Failed to read object header at address: " + address, e);
		}
	}

	public <T extends H5Message> List<T> getMessagesOfType(Class<T> type) {
		return getMessages().stream().filter(type::isInstance).map(type::cast).collect(Collectors.toList());
	}

	public <T extends H5Message> boolean hasMessageOfType(Class<T> type) {
		return !getMessagesOfType(type).isEmpty();
	}

	public <T extends H5Message> T getMessageOfType(Class<T> type) {
		List<T> messagesOfType = getMessagesOfType(type);
		// Validate only one message exists
		if(messagesOfType.isEmpty()) {
			throw new H5RuntimeException("Requested message type '" + type.getSimpleName() + "' not present");
		}
		if(messagesOfType.size() > 1) {
			throw new H5RuntimeException("Requested message type '" + type.getSimpleName() + "' is not unique");
		}

		return messagesOfType.get(0);
	}

	private void readMessages(ByteBuffer bb, int numberOfMessages) {
		while(bb.remaining() > 4 && messages.size() < numberOfMessages) {
			H5Message m = H5Message.readObjectHeaderMessage(rootObject, bb);
			messages.add(m);
		}

	}

	public H5ObjectHeader(H5RootObject rootObject, String datasetName) {
		this.rootObject = rootObject;
		this.datasetName = datasetName;
	}

	private void writeObjectHeader(H5BufferBuilder bb, short numberOfMessages, int headerSize) {
		// Version
		bb.writeByte(rootObject.objectHeaderVersion);

		// Skip reserved byte
		bb.writeByte(0);

		// Number of messages
		bb.writeShort(numberOfMessages);

		// Reference Count
		bb.writeInt(this.referenceCount);

		// Size of the messages
		bb.writeInt(headerSize);

		bb.writeInt(0);
	}

	public H5BufferBuilder toBuffer() {
		H5BufferBuilder bb = new H5BufferBuilder();
		this.toBuffer(bb);
		return bb;
	}

	public void toBuffer(H5BufferBuilder bb) {

		// 1. Write Object Header Message for first step
		this.writeObjectHeader(bb, (short) 1, 24);
		//1.1 Write SymbolTableMessage
		BitSet flags = new BitSet();
		long localHeapAddress = 680;
		H5SymbolTableMessage symbolTableMessage = new H5SymbolTableMessage(rootObject, flags, 136, localHeapAddress);
		symbolTableMessage.toBuffer(bb);

		//1.2 Write BTree data
		List<Long> childAddresses = new ArrayList<>();
		childAddresses.add(1072L);
		H5BTree bTree = new H5BTree(rootObject, (byte) 0, (byte) 0, 1, -1, -1, childAddresses);
		bTree.toBuffer(bb);

		// and static value!
		bb.writeShort((short) 8);
		//long l = bb.getSize();

		// 1.3 Write LocalHeap
		bb.goToPositionWithWriteZero(localHeapAddress);
		H5LocalHeap localHeap = new H5LocalHeap(rootObject, datasetName, 88, 16, 712);
		localHeap.toBuffer(bb);

		// 2. Write Object Header Message for second step
		this.writeObjectHeader(bb, (short) 6, 256);

		// 2.1 Write Data Space
		flags = new BitSet(8);
		//flags.set(0);
		H5DataSpaceMessage dataSpaceMessage = new H5DataSpaceMessage(rootObject, flags);
		dataSpaceMessage.toBuffer(bb);

		flags.set(0);
		// 2.2 Write Data Type
		H5DoubleDataType doubleDataType = new H5DoubleDataType();
		H5DataTypeMessage dataTypeMessage = new H5DataTypeMessage(rootObject, flags, doubleDataType);
		dataTypeMessage.toBuffer(bb);

		// 2.3 Write Fill Value

		H5FillValueMessage fillValueMessage = new H5FillValueMessage(rootObject, flags, 2, 2, true);
		fillValueMessage.toBuffer(bb);

		// 2.4 Write Data Layout Message
		flags = new BitSet();

		H5DataLayoutMessage dataLayoutMessage = new H5DataLayoutMessage(rootObject, flags, 2048,
			(rootObject.row * rootObject.col) * rootObject.superblock.sizeOfLengths);
		dataLayoutMessage.toBuffer(bb);
		// 2.5 Write Object Modification Time
		long time = Instant.now().getEpochSecond();
		H5ObjectModificationTimeMessage objectModificationTimeMessage = new H5ObjectModificationTimeMessage(rootObject,
			flags, time);
		objectModificationTimeMessage.toBuffer(bb);
		//2.6 Write Nil
		H5NilMessage nilMessage = new H5NilMessage(rootObject, flags);
		nilMessage.toBuffer(bb);

		// Write Group Symbol Table Node
		int i = 0;
		for(long child : childAddresses) {

			bb.goToPositionWithWriteZero(child);

			H5SymbolTableEntry[] symbolTableEntries = new H5SymbolTableEntry[1];
			symbolTableEntries[i++] = new H5SymbolTableEntry(8, 800, 0, -1, -1);
			H5GroupSymbolTableNode groupSTE = new H5GroupSymbolTableNode(rootObject, (short) 1, symbolTableEntries);
			groupSTE.toBuffer(bb);
		}
	}

	public List<H5Message> getMessages() {
		return messages;
	}
}
