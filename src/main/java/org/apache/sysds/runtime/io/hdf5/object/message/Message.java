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


package org.apache.sysds.runtime.io.hdf5.object.message;

import org.apache.sysds.runtime.io.hdf5.BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.Superblock;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class Message {
	// Message flags
	private static final int MESSAGE_DATA_CONSTANT = 0;
	private static final int MESSAGE_SHARED = 1;
	private static final int MESSAGE_SHOULD_NOT_BE_SHARED = 2;
	private static final int FAIL_ON_UNKNOWN_MESSAGE_TYPE_WITH_WRITE = 3;
	private static final int SET_FLAG_ON_MODIFICATION_WITH_UNKNOWN_MESSAGE = 4;
	private static final int OBJECT_MODIFIED_WITHOUT_UNDERSTANDING_MESSAGE = 5;
	private static final int MESSAGE_CAN_BE_SHARED = 6;
	private static final int ALWAYS_FAIL_ON_UNKNOWN_MESSAGE_TYPE = 7;

	private final BitSet flags;

	public Message(BitSet flags) {
		this.flags = flags;
	}

	public static Message readObjectHeaderV1Message(ByteBuffer bb, Superblock sb) {
		Utils.seekBufferToNextMultipleOfEight(bb);
		int messageType = Utils.readBytesAsUnsignedInt(bb, 2);
		int dataSize = Utils.readBytesAsUnsignedInt(bb, 2);
		BitSet flags = BitSet.valueOf(new byte[] {bb.get()});

		// Skip 3 reserved zero bytes
		bb.position(bb.position() + 3);

		// Create a new buffer holding this header data
		final ByteBuffer headerData = Utils.createSubBuffer(bb, dataSize);

		final Message message = readMessage(headerData, sb, messageType, flags);
		return message;
	}

	public static BufferBuilder writeObjectHeaderV1Message(BufferBuilder bb, int messageType, Message message) {
		bb.seekBufferBuilderToNextMultipleOfEight();

		// Message Type
		bb.writeShort((short) messageType);
		BufferBuilder bufferBuilder = null;

		switch(messageType) {
			case 1: // 0x0001
				DataSpaceMessage dsm=(DataSpaceMessage) message;
				bufferBuilder = dsm.toBuffer(bb);
				break;

			case 17: // 0x0011
				SymbolTableMessage stm = (SymbolTableMessage) message;
				bufferBuilder = stm.toBuffer(bb);
				break;
		}

		return bufferBuilder;
	}



	public static Message readObjectHeaderV2Message(ByteBuffer bb, Superblock sb) {
		return readObjectHeaderV2Message(bb, sb, false);
	}

	public static Message readObjectHeaderV2Message(ByteBuffer bb, Superblock sb,
		boolean attributeCreationOrderTracked) {
		int messageType = Utils.readBytesAsUnsignedInt(bb, 1);
		int dataSize = Utils.readBytesAsUnsignedInt(bb, 2);
		BitSet flags = BitSet.valueOf(new byte[] {bb.get()});
		if(attributeCreationOrderTracked) {
			//skip creation order
			bb.getShort();
		}

		// Create a new buffer holding this header data
		final ByteBuffer headerData = Utils.createSubBuffer(bb, dataSize);

		final Message message = readMessage(headerData, sb, messageType, flags);

		return message;
	}

	private static Message readMessage(ByteBuffer bb, Superblock sb, int messageType, BitSet flags) {
		System.out.println("message type= "+messageType);
		switch(messageType) {
			case 0: // 0x0000
				return new NilMessage(bb, flags);
			case 1: // 0x0001
				return new DataSpaceMessage(bb, sb, flags);
			case 2: // 0x0002
				return new LinkInfoMessage(bb, sb, flags);
			case 3: // 0x0003
				return new DataTypeMessage(bb, flags);
			case 4: // 0x0004
				return new FillValueOldMessage(bb, flags);
			case 5: // 0x0005
				return new FillValueMessage(bb, flags);
			case 6: // 0x0006
				return new LinkMessage(bb, sb, flags);
			case 8: // 0x0008
				return DataLayoutMessage.createDataLayoutMessage(bb, sb, flags);
			case 9: // 0x0009
				throw new HdfException("Encountered Bogus message. Is this a valid HDF5 file?");
			case 10: // 0x000A
				return new GroupInfoMessage(bb, flags);
			case 11: // 0x000B
				return new FilterPipelineMessage(bb, flags);
			case 12: // 0x000C
				return new AttributeMessage(bb, sb, flags);
			case 13: // 0x000D
				return new ObjectCommentMessage(bb, flags);
			case 14: // 0x000E
				return new OldObjectModificationTimeMessage(bb, flags);
			case 16: // 0x0010
				return new ObjectHeaderContinuationMessage(bb, sb, flags);
			case 17: // 0x0011
				return new SymbolTableMessage(bb, sb, flags);
			case 18: // 0x0012
				return new ObjectModificationTimeMessage(bb, flags);
			case 19: // 0x0013
				return new BTreeKValuesMessage(bb, flags);
			case 20: // 0x0014
				throw new UnsupportedHdfException("Encountered Driver Info Message, this is not supported by jHDF");
			case 21: // 0x0015
				return new AttributeInfoMessage(bb, sb, flags);
			case 22: // 0x0016
				return new ObjectReferenceCountMessage(bb, flags);

			default:
				throw new HdfException("Unrecognized message type = " + messageType);
		}
	}

	public boolean isMessageDataConstant() {
		return flags.get(MESSAGE_DATA_CONSTANT);
	}

	public boolean isMessageShared() {
		return flags.get(MESSAGE_SHARED);
	}

	public boolean isMessageNotShared() {
		return flags.get(MESSAGE_SHOULD_NOT_BE_SHARED);
	}

	public boolean isFailOnUnknownTypeWithWrite() {
		return flags.get(FAIL_ON_UNKNOWN_MESSAGE_TYPE_WITH_WRITE);
	}

	public boolean isFlagToBeSetOnUnknownType() {
		return flags.get(SET_FLAG_ON_MODIFICATION_WITH_UNKNOWN_MESSAGE);
	}

	public boolean isObjectModifiedWithoutUnderstandingOfThisMessage() {
		return flags.get(OBJECT_MODIFIED_WITHOUT_UNDERSTANDING_MESSAGE);
	}

	public boolean isMessageShareable() {
		return flags.get(MESSAGE_CAN_BE_SHARED);
	}

	public boolean isAlwaysFailOnUnknownType() {
		return flags.get(ALWAYS_FAIL_ON_UNKNOWN_MESSAGE_TYPE);
	}

	public BitSet getFlags() {
		return this.flags;
	}

}
