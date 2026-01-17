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

package org.apache.sysds.runtime.io.hdf5.message;

import org.apache.sysds.runtime.io.hdf5.H5BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.H5Constants;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.H5RuntimeException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public abstract class H5Message {

	private final BitSet flags;

	protected H5RootObject rootObject;

	public H5Message(H5RootObject rootObject, BitSet flags) {
		this.rootObject = rootObject;
		this.flags = flags;
	}

	public static H5Message readObjectHeaderMessage(H5RootObject rootObject, ByteBuffer bb) {
		Utils.seekBufferToNextMultipleOfEight(bb);
		int messageType = Utils.readBytesAsUnsignedInt(bb, 2);
		int dataSize = Utils.readBytesAsUnsignedInt(bb, 2);
		BitSet flags = BitSet.valueOf(new byte[] {bb.get()});

		// Skip 3 reserved zero bytes
		bb.position(bb.position() + 3);

		// Create a new buffer holding this header data
		final ByteBuffer headerData = Utils.createSubBuffer(bb, dataSize);
		final H5Message message = readMessage(rootObject, headerData, messageType, flags);
		return message;
	}

	protected H5BufferBuilder toBuffer() {
		H5BufferBuilder bb = new H5BufferBuilder();
		this.toBuffer(bb);
		return bb;
	}

	protected void toBuffer(H5BufferBuilder bb){
		throw new H5RuntimeException("Unimplemented method!");
	}

	protected void toBuffer(H5BufferBuilder bb, int messageType) {

		// Message Type
		bb.writeShort((short) messageType);
		byte[] reserved = {(byte) 0, 0, 0};

		switch(messageType) {
			case H5Constants.NIL_MESSAGE:
				// Data Size
				bb.writeShort((short) 104);

				break;
			case H5Constants.DATA_SPACE_MESSAGE:
				// Data Size
				bb.writeShort((short) 40);

				break;
			case H5Constants.DATA_TYPE_MESSAGE:
				// Data Size
				bb.writeShort((short) 24);

				break;
			case H5Constants.FILL_VALUE_MESSAGE:
				// Data Size
				bb.writeShort((short) 8);

				break;
			case H5Constants.SYMBOL_TABLE_MESSAGE:
				// Data Size
				bb.writeShort((short) 16);

				break;
			case H5Constants.OBJECT_MODIFICATION_TIME_MESSAGE:
				// Data Size
				bb.writeShort((short) 8);
				break;
			case H5Constants.DATA_LAYOUT_MESSAGE:
				// Data Size
				bb.writeShort((short) 24);
				break;
			default:
				throw new H5RuntimeException("Unrecognized message type = " + messageType);
		}
		// Flags
		if(flags.length() != 0) {
			bb.writeBitSet(flags, flags.length());
		}
		else {
			bb.writeByte(0);
		}

		// Skip 3 reserved zero bytes
		bb.writeBytes(reserved);
	}

	private static H5Message readMessage(H5RootObject rootObject, ByteBuffer bb, int messageType, BitSet flags) {
		switch(messageType) {
			case H5Constants.NIL_MESSAGE:
				return new H5NilMessage(rootObject, flags, bb);

			case H5Constants.DATA_SPACE_MESSAGE:
				return new H5DataSpaceMessage(rootObject, flags, bb);

			case H5Constants.DATA_TYPE_MESSAGE:
				return new H5DataTypeMessage(rootObject, flags, bb);

			case H5Constants.FILL_VALUE_MESSAGE:
				return new H5FillValueMessage(rootObject, flags, bb);

			case H5Constants.DATA_LAYOUT_MESSAGE:
				return new H5DataLayoutMessage(rootObject, flags, bb);

			case H5Constants.SYMBOL_TABLE_MESSAGE:
				return new H5SymbolTableMessage(rootObject, flags, bb);

			case H5Constants.OBJECT_MODIFICATION_TIME_MESSAGE:
				return new H5ObjectModificationTimeMessage(rootObject, flags, bb);

			case H5Constants.FILTER_PIPELINE_MESSAGE:
				return new H5FilterPipelineMessage(rootObject, flags, bb);

			case H5Constants.ATTRIBUTE_MESSAGE:
				return new H5AttributeMessage(rootObject, flags, bb);

			default:
				throw new H5RuntimeException("Unrecognized message type = " + messageType);
		}
	}
}
