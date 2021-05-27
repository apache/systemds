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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class SymbolTableMessage extends Message {

	private final long bTreeAddress;
	private final long localHeapAddress;

	SymbolTableMessage(ByteBuffer bb, Superblock sb, BitSet flags) {
		super(flags);
		bTreeAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
		localHeapAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
	}

	public SymbolTableMessage(BitSet flags, long bTreeAddress, long localHeapAddress) {
		super(flags);
		this.bTreeAddress = bTreeAddress;
		this.localHeapAddress = localHeapAddress;
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}
	public BufferBuilder toBuffer(BufferBuilder header) {

		// Data Size: 16 two long values
		header.writeShort((short) 16);

		// Flags
		// TODO: read about the flags
		if(this.getFlags().length()!=0) {
			header.writeBitSet(this.getFlags(), this.getFlags().length());
		}
		else {
			header.writeByte(0);
		}

		// Skip 3 reserved zero bytes
		byte[] reserved={(byte) 0,0,0};
		header.writeBytes(reserved);

		// Write values
		header.writeLong(this.bTreeAddress);
		header.writeLong(this.localHeapAddress);

		return header;
	}

	public long getbTreeAddress() {
		return bTreeAddress;
	}

	public long getLocalHeapAddress() {
		return localHeapAddress;
	}


}
