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
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class FillValueMessage extends Message {

	private static final int FILL_VALUE_DEFINED_BIT = 5;

	private final int spaceAllocationTime;
	private final int fillValueWriteTime;
	private final boolean fillValueDefined;
	private final ByteBuffer fillValue;
	private byte version;

	public FillValueMessage(byte version, BitSet flags, int spaceAllocationTime, int fillValueWriteTime, boolean fillValueDefined) {
		super(flags);
		this.version=version;
		this.spaceAllocationTime = spaceAllocationTime;
		this.fillValueWriteTime = fillValueWriteTime;
		this.fillValueDefined = fillValueDefined;
		this.fillValue=null;
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}

	public BufferBuilder toBuffer(BufferBuilder header) {
		header.writeByte(version);
		if (version == 1 || version == 2) {
			header.writeByte(spaceAllocationTime);
			header.writeByte(fillValueWriteTime);
			if(fillValueDefined){
				header.writeByte(1);
			}
			else
				header.writeByte(0);

//			if (version == 2 && fillValueDefined){
//				//TODO
//			}
		}

		return header;
	}

	FillValueMessage(ByteBuffer bb, BitSet messageFlags) {
		super(messageFlags);

		final byte version = bb.get();
		if (version == 1 || version == 2) {
			spaceAllocationTime = bb.get();
			fillValueWriteTime = bb.get();
			boolean fillValueMaybeDefined = bb.get() == 1;

			if (version == 2 && fillValueMaybeDefined) {
				int size = Utils.readBytesAsUnsignedInt(bb, 4);
				if (size > 0) {
					fillValue = Utils.createSubBuffer(bb, size);
					fillValueDefined = true;
				} else {
					fillValue = null;
					fillValueDefined = false;
				}
			} else {
				fillValue = null; // No fill value defined
				fillValueDefined = false;
			}
		} else if (version == 3) {
			BitSet flags = BitSet.valueOf(new byte[] { bb.get() });
			spaceAllocationTime = Utils.bitsToInt(flags, 0, 2); // 0-1
			fillValueWriteTime = Utils.bitsToInt(flags, 2, 2); // 2-3
			fillValueDefined = flags.get(FILL_VALUE_DEFINED_BIT);

			if (fillValueDefined) {
				int size = Utils.readBytesAsUnsignedInt(bb, 4);
				fillValue = Utils.createSubBuffer(bb, size);
			} else {
				fillValue = null; // No fill value defined
			}
		} else {
			throw new HdfException("Unrecognized version = " + version);
		}
	}

	public boolean isFillValueDefined() {
		return fillValueDefined;
	}

	public int getSpaceAllocationTime() {
		return spaceAllocationTime;
	}

	public int getFillValueWriteTime() {
		return fillValueWriteTime;
	}

	public ByteBuffer getFillValue() {
		return fillValue.asReadOnlyBuffer();
	}
}
