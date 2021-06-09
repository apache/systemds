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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class H5FillValueMessage extends H5Message {

	private final int spaceAllocationTime;
	private final int fillValueWriteTime;
	private final boolean fillValueDefined;
	private final ByteBuffer fillValue;

	public H5FillValueMessage(H5RootObject rootObject, BitSet flags, int spaceAllocationTime, int fillValueWriteTime,
		boolean fillValueDefined) {

		super(rootObject, flags);
		this.spaceAllocationTime = spaceAllocationTime;
		this.fillValueWriteTime = fillValueWriteTime;
		this.fillValueDefined = fillValueDefined;
		this.fillValue = null;
	}

	public H5FillValueMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);

		// version
		rootObject.setFillValueVersion(bb.get());

		spaceAllocationTime = bb.get();
		fillValueWriteTime = bb.get();
		boolean fillValueMaybeDefined = bb.get() == 1;

		if(fillValueMaybeDefined) {
			int size = Utils.readBytesAsUnsignedInt(bb, 4);
			if(size > 0) {
				fillValue = Utils.createSubBuffer(bb, size);
				fillValueDefined = true;
			}
			else {
				fillValue = null;
				fillValueDefined = false;
			}
		}
		else {
			fillValue = null;
			fillValueDefined = false;
		}
	}

	@Override
	public void toBuffer(H5BufferBuilder bb) {
		super.toBuffer(bb, H5Constants.FILL_VALUE_MESSAGE);
		bb.writeByte(rootObject.getFillValueVersion());
		bb.writeByte(spaceAllocationTime);
		bb.writeByte(fillValueWriteTime);
		if(fillValueDefined) {
			bb.writeByte(1);
		}
		else
			bb.writeByte(0);
		// Reserve 4 byte
		bb.writeInt(0);

	}

	public int getSpaceAllocationTime() {
		return spaceAllocationTime;
	}

	public int getFillValueWriteTime() {
		return fillValueWriteTime;
	}

	public boolean isFillValueDefined() {
		return fillValueDefined;
	}

	public ByteBuffer getFillValue() {
		return fillValue;
	}
}
