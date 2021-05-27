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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class DataSpaceMessage extends Message {

	private final DataSpace dataSpace;

	public DataSpaceMessage(ByteBuffer bb, Superblock sb, BitSet flags) {
		super(flags);
		dataSpace = DataSpace.readDataSpace(bb, sb);
	}

	public  DataSpaceMessage(byte version, int numberOfDimensions, int[] dimensions, int[] maxSizes,
		int sizeOfLengths){
		super(new BitSet());
		dataSpace = new DataSpace(version, numberOfDimensions, dimensions, maxSizes, sizeOfLengths);
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}
	public BufferBuilder toBuffer(BufferBuilder header) {
		return dataSpace.toBuffer(header);
	}


	public DataSpace getDataSpace() {
		return dataSpace;
	}
}
