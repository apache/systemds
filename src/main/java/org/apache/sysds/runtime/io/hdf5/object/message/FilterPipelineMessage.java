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

import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;
import org.apache.commons.lang3.ArrayUtils;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

public class FilterPipelineMessage extends Message {

	private static final int OPTIONAL = 0;

	private final List<FilterInfo> filters;

	public FilterPipelineMessage(ByteBuffer bb, BitSet messageFlags) {
		super(messageFlags);

		final byte version = bb.get();

		if (version != 1 && version != 2) {
			throw new UnsupportedHdfException("Only filer pipeline version 1 or 2 are supported");
		}

		final byte numberOfFilters = bb.get();
		filters = new ArrayList<>(numberOfFilters);

		if (version == 1) {
			// Skip 6 reserved bytes
			bb.position(bb.position() + 6);
		}

		// Read filters
		for (int i = 0; i < numberOfFilters; i++) {
			// FilterInfo ID
			final int filterId = Utils.readBytesAsUnsignedInt(bb, 2);

			// Name length
			final int nameLength;
			if (version == 2 && filterId < 256) {
				nameLength = 0;
			} else {
				nameLength = Utils.readBytesAsUnsignedInt(bb, 2);
			}

			// 2 bytes of optional
			final BitSet flags = BitSet.valueOf(new byte[] { bb.get(), bb.get() });
			final boolean optional = flags.get(OPTIONAL);

			final int numberOfDataValues = Utils.readBytesAsUnsignedInt(bb, 2);

			final String name;
			if (nameLength >= 2) {
				name = Utils.readUntilNull(Utils.createSubBuffer(bb, nameLength));
			} else {
				name = "undefined";
			}

			final int[] data = new int[numberOfDataValues];
			for (int j = 0; j < numberOfDataValues; j++) {
				data[j] = bb.getInt();
			}
			// If there are a odd number of values then there are 4 bytes of padding
			if (version == 1 && numberOfDataValues % 2 != 0) {
				// Skip 4 padding bytes
				bb.position(bb.position() + 4);
			}

			filters.add(new FilterInfo(filterId, name, optional, data));
		}

	}

	public List<FilterInfo> getFilters() {
		return filters;
	}

	public static class FilterInfo {

		private final int id;
		private final String name;
		private final boolean optional;
		private final int[] data;

		public FilterInfo(int id, String name, boolean optional, int[] data) {
			this.id = id;
			this.name = name;
			this.optional = optional;
			this.data = ArrayUtils.clone(data);
		}

		public int getId() {
			return id;
		}

		public String getName() {
			return name;
		}

		public boolean isOptional() {
			return optional;
		}

		public int[] getData() {
			return ArrayUtils.clone(data);
		}

		@Override
		public String toString() {
			return "FilterInfo [id=" + id + ", name=" + name + ", optional=" + optional + ", data="
					+ Arrays.toString(data)
					+ "]";
		}
	}
}
