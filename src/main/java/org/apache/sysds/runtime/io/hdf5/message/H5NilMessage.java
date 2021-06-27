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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class H5NilMessage extends H5Message {

	public H5NilMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);
		// Move buffer to the end
		bb.position(bb.limit());
	}

	public H5NilMessage(H5RootObject rootObject, BitSet flags) {
		super(rootObject, flags);
	}

	@Override
	public void toBuffer(H5BufferBuilder bb) {
		super.toBuffer(bb, H5Constants.NIL_MESSAGE);
		byte[] reserve = new byte[104];
		bb.writeBytes(reserve);
	}
}
