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


package org.apache.sysds.runtime.io.hdf5.dataset;

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.object.datatype.CompoundDataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.CompoundDataType.CompoundDataMember;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class CompoundDatasetReader {

	private CompoundDatasetReader() {
		throw new AssertionError("No instances of CompoundDatasetReader");
	}

	public static Map<String, Object> readDataset(CompoundDataType type, ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		final int sizeAsInt = Arrays.stream(dimensions).reduce(1, Math::multiplyExact);

		final List<CompoundDataMember> members = type.getMembers();

		final Map<String, Object> data = new LinkedHashMap<>(members.size());

		for (CompoundDataMember member : members) {
			final byte[] memberBytes = new byte[member.getDataType().getSize()];
			final ByteBuffer memberBuffer = ByteBuffer.allocate(member.getDataType().getSize() * sizeAsInt);

			// Loop through the date buffer extracting the bytes for this member
			for (int i = 0; i < sizeAsInt; i++) {
				buffer.position(type.getSize() * i + member.getOffset());
				buffer.get(memberBytes, 0, memberBytes.length);
				memberBuffer.put(memberBytes);
			}

			// Now read this member
			memberBuffer.rewind();

			final Object memberData = DatasetReader.readDataset(member.getDataType(), memberBuffer, dimensions, hdfFc);
			data.put(member.getName(), memberData);
		}

		return data;
	}
}
