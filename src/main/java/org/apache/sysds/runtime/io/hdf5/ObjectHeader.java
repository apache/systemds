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

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.hdf5.btree.BTreeV1Group;
import org.apache.sysds.runtime.io.hdf5.checksum.ChecksumUtils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.object.message.*;
import org.apache.sysds.runtime.io.hdf5.object.message.ObjectHeaderContinuationMessage;
import org.apache.commons.lang3.concurrent.LazyInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.io.hdf5.Utils.readBytesAsUnsignedInt;

public abstract class ObjectHeader {
	private static final Logger logger = LoggerFactory.getLogger(ObjectHeader.class);

	private final long address;

	protected final List<Message> messages = new ArrayList<>();

	public long getAddress() {
		return address;
	}

	public abstract int getVersion();

	public abstract boolean isAttributeCreationOrderTracked();

	public abstract boolean isAttributeCreationOrderIndexed();

	public List<Message> getMessages() {
		return messages;
	}

	public ObjectHeader(long address) {
		this.address = address;
	}

	public <T extends Message> List<T> getMessagesOfType(Class<T> type) {
		return getMessages().stream().filter(type::isInstance).map(type::cast).collect(Collectors.toList());
	}

	public <T extends Message> boolean hasMessageOfType(Class<T> type) {
		return !getMessagesOfType(type).isEmpty();
	}

	public <T extends Message> T getMessageOfType(Class<T> type) {
		List<T> messagesOfType = getMessagesOfType(type);
		// Validate only one message exists
		if(messagesOfType.isEmpty()) {
			throw new HdfException("Requested message type '" + type.getSimpleName() + "' not present");
		}
		if(messagesOfType.size() > 1) {
			throw new HdfException("Requested message type '" + type.getSimpleName() + "' is not unique");
		}

		return messagesOfType.get(0);
	}

	public static class ObjectHeaderV1 extends ObjectHeader {

		private final byte version;
		private final int referenceCount;

		private int numberOfDimensions;
		private int[] dimensions;
		private int[] maxSizes;
		private int sizeOfLengths;
		private int sizeOfOffsets;
		private String childName;

		private ObjectHeaderV1(HdfFileChannel hdfFc, long address) {
			super(address);
			System.out.println("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   "+ address  );
			try {
				ByteBuffer header = hdfFc.readBufferFromAddress(address, 12);
				// Version
				version = header.get();
				if(version != 1) {
					throw new HdfException("Invalid version detected. Version is = " + version);
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
				header = hdfFc.readBufferFromAddress(address, headerSize);

				System.out.println("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE   "+ address  );
				System.out.println("numberOfMessages = "+ numberOfMessages + "   headerSize= "+headerSize);

				readMessages(hdfFc, header, numberOfMessages);

			}
			catch(Exception e) {
				throw new HdfException("Failed to read object header at address: " + address, e);
			}
		}

		private void readMessages(HdfFileChannel hdfFc, ByteBuffer bb, int numberOfMessages) {
			while(bb.remaining() > 4 && messages.size() < numberOfMessages) {
				Message m = Message.readObjectHeaderV1Message(bb, hdfFc.getSuperblock());
				messages.add(m);

				Gson gson = new Gson();
				System.out.println(gson.toJson(m));
				System.out.println("++++++++++++++++++++++++++++++++");

				if(m instanceof ObjectHeaderContinuationMessage) {
					ObjectHeaderContinuationMessage ohcm = (ObjectHeaderContinuationMessage) m;

					ByteBuffer continuationBuffer = hdfFc.readBufferFromAddress(ohcm.getOffset(), ohcm.getLength());

					readMessages(hdfFc, continuationBuffer, numberOfMessages);
				}
			}
		}

		public ObjectHeaderV1(byte version, int referenceCount, int numberOfDimensions, int[] dimensions,
			int[] maxSizes, int sizeOfLengths, int sizeOfOffsets, String childName) {
			super(0);
			this.version = version;
			this.referenceCount = referenceCount;
			this.numberOfDimensions = numberOfDimensions;
			this.dimensions = dimensions;
			this.maxSizes = maxSizes;
			this.sizeOfLengths = sizeOfLengths;
			this.sizeOfOffsets = sizeOfOffsets;
			this.childName = childName;

		}

		public BufferBuilder toBuffer() {
			BufferBuilder header = new BufferBuilder();
			return toBuffer(header);
		}

		public BufferBuilder toBuffer(BufferBuilder header) {

			if(version != 1) {
				throw new HdfException("Invalid version detected. Version is = " + version);
			}
			// Version
			header.writeByte(this.version);

			// Skip reserved byte
			header.writeByte(0);

			// Number of messages
			short numberOfMessages = 1;
			header.writeShort(numberOfMessages);

			// Reference Count
			header.writeInt(this.referenceCount);

			// Size of the messages
			int headerSize = 24;
			header.writeInt(headerSize);

			// Write SymbolTableMessage
			BitSet flags = new BitSet();
			long localHeapAddress = 680;
			SymbolTableMessage symbolTableMessage = new SymbolTableMessage(flags, 136, localHeapAddress);
			Message.writeObjectHeaderV1Message(header, Constants.SYMBOL_TABLE_MESSAGE, symbolTableMessage);

			List<Long> childAddresses = new ArrayList<>();
			childAddresses.add(1072L);
			BTreeV1Group.BTreeV1GroupLeafNode bTreeV1GroupLeafNode = new BTreeV1Group.BTreeV1GroupLeafNode((byte) 0,
				(byte) 0, 1, -1, -1, sizeOfLengths, sizeOfOffsets, childAddresses);
			bTreeV1GroupLeafNode.toBuffer(header);

			header.goToPositionWithWriteZero(localHeapAddress);
			LocalHeap localHeap = new LocalHeap((short) 0, childName, 88, 16, 712);
			localHeap.toBuffer(header);

			// second header
			// Version
			header.writeByte(this.version);

			// Skip reserved byte
			header.writeByte(0);

			// Number of messages
			numberOfMessages = 6;
			header.writeShort(numberOfMessages);

			// Reference Count
			header.writeInt(this.referenceCount);

			// Size of the messages
			headerSize = 256;
			header.writeInt(headerSize);

			DataSpaceMessage dataSpaceMessage = new DataSpaceMessage((byte) 1, numberOfDimensions, dimensions, maxSizes,
				sizeOfLengths);
			Message.writeObjectHeaderV1Message(header, Constants.DATA_SPACE_MESSAGE, dataSpaceMessage);

			flags = new BitSet(8);
			flags.set(0);
			DataTypeMessage dataTypeMessage=new DataTypeMessage(Constants.FLOAT_POINT,flags);
			Message.writeObjectHeaderV1Message(header, Constants.DATA_TYPE_MESSAGE, dataTypeMessage);

			FillValueMessage fillValueMessage=new FillValueMessage((byte) 2, flags, 2,2,false);
			Message.writeObjectHeaderV1Message(header, Constants.FILL_VALUE_MESSAGE, fillValueMessage);

			//----------------------

			for(long child : childAddresses) {
				header.goToPositionWithWriteZero(child);

				SymbolTableEntry[] symbolTableEntries = new SymbolTableEntry[1];
				symbolTableEntries[0] = new SymbolTableEntry(8, 800, 0, -1, -1, -1);
				GroupSymbolTableNode groupSTE = new GroupSymbolTableNode((byte) 1, (short) 1, symbolTableEntries);

				groupSTE.toBuffer(header);
			}

			//			DataSpaceMessage dataSpaceMessage = new DataSpaceMessage((byte) 1, numberOfDimensions, dimensions, maxSizes,
			//				sizeOfLengths);
			//			Message.writeObjectHeaderV1Message(header, Constants.DATA_SPACE_MESSAGE, dataSpaceMessage);

			return header;
		}

		@Override public int getVersion() {
			return version;
		}

		public int getReferenceCount() {
			return referenceCount;
		}

		@Override public boolean isAttributeCreationOrderTracked() {
			return false; // Not supported in v1 headers
		}

		@Override public boolean isAttributeCreationOrderIndexed() {
			return false; // Not supported in v1 headers
		}

	}

	public static class ObjectHeaderV2 extends ObjectHeader {

		private static final byte[] OBJECT_HEADER_V2_SIGNATURE = "OHDR".getBytes(StandardCharsets.US_ASCII);
		private static final byte[] OBJECT_HEADER_V2_CONTINUATION_SIGNATURE = "OCHK"
			.getBytes(StandardCharsets.US_ASCII);

		private static final int ATTRIBUTE_CREATION_ORDER_TRACKED = 2;
		private static final int ATTRIBUTE_CREATION_ORDER_INDEXED = 3;
		private static final int NUMBER_OF_ATTRIBUTES_PRESENT = 4;
		private static final int TIMESTAMPS_PRESENT = 5;

		/**
		 * Type of node. 0 = group, 1 = data
		 */
		private final byte version;

		private final long accessTime;
		private final long modificationTime;
		private final long changeTime;
		private final long birthTime;

		private final int maximumNumberOfCompactAttributes;
		private final int maximumNumberOfDenseAttributes;
		private final BitSet flags;

		private ObjectHeaderV2(HdfFileChannel hdfFc, long address) {
			super(address);
			int headerSize = 0; // Keep track of the size for checksum

			try {
				ByteBuffer bb = hdfFc.readBufferFromAddress(address, 6);
				address += 6;
				headerSize += 6;

				byte[] formatSignatureBytes = new byte[OBJECT_HEADER_V2_SIGNATURE.length];
				bb.get(formatSignatureBytes);

				// Verify signature
				if(!Arrays.equals(OBJECT_HEADER_V2_SIGNATURE, formatSignatureBytes)) {
					throw new HdfException("Object header v2 signature not matched");
				}

				// Version
				version = bb.get();

				if(version != 2) {
					throw new HdfException("Invalid version detected. Version is = " + version);
				}

				// Flags
				flags = BitSet.valueOf(new byte[] {bb.get()});

				// Size of chunk 0
				final byte sizeOfChunk0;
				if(flags.get(1)) {
					if(flags.get(0)) {
						sizeOfChunk0 = 8;
					}
					else {
						sizeOfChunk0 = 4;
					}
				}
				else { // bit 0 = false
					if(flags.get(0)) {
						sizeOfChunk0 = 2;
					}
					else {
						sizeOfChunk0 = 1;
					}
				}

				// Timestamps
				if(flags.get(TIMESTAMPS_PRESENT)) {
					bb = hdfFc.readBufferFromAddress(address, 16);
					address += 16;
					headerSize += 16;

					accessTime = Utils.readBytesAsUnsignedLong(bb, 4);
					modificationTime = Utils.readBytesAsUnsignedLong(bb, 4);
					changeTime = Utils.readBytesAsUnsignedLong(bb, 4);
					birthTime = Utils.readBytesAsUnsignedLong(bb, 4);
				}
				else {
					accessTime = -1;
					modificationTime = -1;
					changeTime = -1;
					birthTime = -1;
				}

				// Number of attributes
				if(flags.get(NUMBER_OF_ATTRIBUTES_PRESENT)) {
					bb = hdfFc.readBufferFromAddress(address, 4);
					address += 4;
					headerSize += 4;

					maximumNumberOfCompactAttributes = readBytesAsUnsignedInt(bb, 2);
					maximumNumberOfDenseAttributes = readBytesAsUnsignedInt(bb, 2);
				}
				else {
					maximumNumberOfCompactAttributes = -1;
					maximumNumberOfDenseAttributes = -1;
				}

				bb = hdfFc.readBufferFromAddress(address, sizeOfChunk0);
				address += sizeOfChunk0;
				headerSize += sizeOfChunk0;

				int sizeOfMessages = readBytesAsUnsignedInt(bb, sizeOfChunk0);

				bb = hdfFc.readBufferFromAddress(address, sizeOfMessages);
				headerSize += sizeOfMessages;

				// There might be a gap at the end of the header of up to 4 bytes
				// message type (1_byte) + message size (2 bytes) + message flags (1 byte)
				readMessages(hdfFc, bb);

				// Checksum
				headerSize += 4;
				ByteBuffer fullHeaderBuffer = hdfFc.readBufferFromAddress(super.getAddress(), headerSize);
				ChecksumUtils.validateChecksum(fullHeaderBuffer);

				logger.debug("Read object header from address: {}", address);

			}
			catch(Exception e) {
				throw new HdfException("Failed to read object header at address: " + address, e);
			}
		}

		private void readMessages(HdfFileChannel hdfFc, ByteBuffer bb) {
			while(bb.remaining() >= 8) {
				Message m = Message
					.readObjectHeaderV2Message(bb, hdfFc.getSuperblock(), this.isAttributeCreationOrderTracked());
				messages.add(m);

				if(m instanceof ObjectHeaderContinuationMessage) {
					ObjectHeaderContinuationMessage ohcm = (ObjectHeaderContinuationMessage) m;
					ByteBuffer continuationBuffer = hdfFc.readBufferFromAddress(ohcm.getOffset(), ohcm.getLength());

					// Verify continuation block signature
					byte[] continuationSignatureBytes = new byte[OBJECT_HEADER_V2_CONTINUATION_SIGNATURE.length];
					continuationBuffer.get(continuationSignatureBytes);
					if(!Arrays.equals(OBJECT_HEADER_V2_CONTINUATION_SIGNATURE, continuationSignatureBytes)) {
						throw new HdfException(
							"Object header continuation header not matched, at address: " + ohcm.getOffset());
					}

					// Recursively read messages
					readMessages(hdfFc, continuationBuffer);

					continuationBuffer.rewind();
					ChecksumUtils.validateChecksum(continuationBuffer);
				}
			}
		}

		@Override public int getVersion() {
			return version;
		}

		public long getAccessTime() {
			return accessTime;
		}

		public long getModificationTime() {
			return modificationTime;
		}

		public long getChangeTime() {
			return changeTime;
		}

		public long getBirthTime() {
			return birthTime;
		}

		public int getMaximumNumberOfCompactAttributes() {
			return maximumNumberOfCompactAttributes;
		}

		public int getMaximumNumberOfDenseAttributes() {
			return maximumNumberOfDenseAttributes;
		}

		@Override public boolean isAttributeCreationOrderTracked() {
			return flags.get(ATTRIBUTE_CREATION_ORDER_TRACKED);
		}

		@Override public boolean isAttributeCreationOrderIndexed() {
			return flags.get(ATTRIBUTE_CREATION_ORDER_INDEXED);
		}

	}

	public static ObjectHeader readObjectHeader(HdfFileChannel hdfFc, long address) {
		ByteBuffer bb = hdfFc.readBufferFromAddress(address, 1);
		byte version = bb.get();
		if(version == 1) {
			return new ObjectHeaderV1(hdfFc, address);
		}
		else {
			return new ObjectHeaderV2(hdfFc, address);
		}
	}

	public static void lazyReadObjectHeader2(HdfFileChannel hdfFc, long address) {
		readObjectHeader(hdfFc, address);
	}

	public static void writeObjectHeader(byte version, BufferBuilder bb, int numberOfDimensions, int[] dimensions,
		int[] maxSizes, int sizeOfLengths, int sizeOfOffsets, String childName) {
		if(version == 0) {
			ObjectHeaderV1 objectHeaderV1 = new ObjectHeaderV1((byte) 1, 1, numberOfDimensions, dimensions, maxSizes,
				sizeOfLengths, sizeOfOffsets, childName);
			objectHeaderV1.toBuffer(bb);
		}
		else {

		}

	}

	public static LazyInitializer<ObjectHeader> lazyReadObjectHeader(HdfFileChannel hdfFc, long address) {

		return new LazyInitializer<ObjectHeader>() {
			@Override protected ObjectHeader initialize() {
				return readObjectHeader(hdfFc, address);
			}
		};
	}
}
