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

import java.nio.ByteBuffer;
import java.util.BitSet;

import org.apache.sysds.runtime.io.hdf5.H5RootObject;

/**
 * Lightweight placeholder for attribute messages. We currently ignore attribute content but keep track of the
 * bytes to ensure the buffer position stays consistent, logging that the attribute was skipped to aid debugging.
 */
public class H5AttributeMessage extends H5Message {

	public H5AttributeMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);
		if(bb.remaining() == 0)
			return;
		byte version = bb.get();
		if(H5RootObject.HDF5_DEBUG) {
			System.out.println("[HDF5] Skipping attribute message v" + version + " (" + bb.remaining() + " bytes payload)");
		}
		// consume the rest of the payload
		bb.position(bb.limit());
	}
}
