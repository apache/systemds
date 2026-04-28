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
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;

import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.io.hdf5.H5RuntimeException;
import org.apache.sysds.runtime.io.hdf5.Utils;

/**
 * Minimal parser for filter pipeline messages. We currently do not support any filters, and therefore
 * fail fast if we encounter one so the user understands why the dataset cannot be read.
 */
public class H5FilterPipelineMessage extends H5Message {

	private final List<Integer> filterIds = new ArrayList<>();

	public H5FilterPipelineMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);
		byte version = bb.get();
		byte numberOfFilters = bb.get();
		// Skip 6 reserved bytes
		bb.position(bb.position() + 6);

		for(int i = 0; i < Byte.toUnsignedInt(numberOfFilters); i++) {
			int filterId = Utils.readBytesAsUnsignedInt(bb, 2);
			int nameLength = Utils.readBytesAsUnsignedInt(bb, 2);
			Utils.readBytesAsUnsignedInt(bb, 2); // flags
			int clientDataLength = Utils.readBytesAsUnsignedInt(bb, 2);

			if(nameLength > 0) {
				byte[] nameBytes = new byte[nameLength];
				bb.get(nameBytes);
			}
			for(int j = 0; j < clientDataLength; j++) {
				Utils.readBytesAsUnsignedInt(bb, 4);
			}
			Utils.seekBufferToNextMultipleOfEight(bb);
			filterIds.add(filterId);
		}

		if(!filterIds.isEmpty()) {
			if(H5RootObject.HDF5_DEBUG) {
				System.out.println("[HDF5] Detected unsupported filter pipeline v" + version + " -> " + filterIds);
			}
			throw new H5RuntimeException("Encountered unsupported filtered dataset (filters=" + filterIds + "). "
				+ "Compressed HDF5 inputs are currently unsupported.");
		}
	}

	public List<Integer> getFilterIds() {
		return Collections.unmodifiableList(filterIds);
	}
}
