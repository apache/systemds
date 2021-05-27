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
import org.apache.sysds.runtime.io.hdf5.ObjectHeader;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.api.ContiguousDataset;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.ContiguousDataLayoutMessage;

import java.nio.ByteBuffer;

import static org.apache.sysds.runtime.io.hdf5.Constants.UNDEFINED_ADDRESS;

public class ContiguousDatasetImpl extends DatasetBase implements ContiguousDataset {

	final ContiguousDataLayoutMessage contiguousDataLayoutMessage;

	public ContiguousDatasetImpl(HdfFileChannel hdfFc, long address, String name, Group parent, ObjectHeader oh) {
		super(hdfFc, address, name, parent, oh);
		this.contiguousDataLayoutMessage = getHeaderMessage(ContiguousDataLayoutMessage.class);
	}

	@Override
	public ByteBuffer getDataBuffer() {
		try {
			ByteBuffer data = hdfFc.map(contiguousDataLayoutMessage.getAddress(), getSizeInBytes());
			convertToCorrectEndiness(data);
			return data;
		} catch (Exception e) {
			throw new HdfException("Failed to map data buffer for dataset '" + getPath() + "'", e);
		}
	}

	@Override
	public ByteBuffer getBuffer() {
		return getDataBuffer();
	}

	@Override
	public long getDataAddress() {
		return contiguousDataLayoutMessage.getAddress();
	}

	@Override
	public boolean isEmpty() {
		return contiguousDataLayoutMessage.getAddress() == UNDEFINED_ADDRESS;
	}
}
