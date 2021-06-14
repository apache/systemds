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


package org.apache.sysds.runtime.io.hdf5;

import org.apache.sysds.runtime.io.hdf5.message.H5DataLayoutMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataSpaceMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataTypeMessage;

import java.nio.ByteBuffer;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5ContiguousDataset {

	private final H5RootObject rootObject;
	private final H5DataLayoutMessage dataLayoutMessage;
	private final H5DataTypeMessage dataTypeMessage;
	@SuppressWarnings("unused")
	private final H5DataSpaceMessage dataSpaceMessage;

	public H5ContiguousDataset(H5RootObject rootObject, H5ObjectHeader objectHeader) {

		this.rootObject = rootObject;
		this.dataLayoutMessage = objectHeader.getMessageOfType(H5DataLayoutMessage.class);
		this.dataTypeMessage = objectHeader.getMessageOfType(H5DataTypeMessage.class);
		this.dataSpaceMessage = objectHeader.getMessageOfType(H5DataSpaceMessage.class);
	}

	public ByteBuffer getDataBuffer(int row) {
		try {
			long rowPos = row * rootObject.getCol()*this.dataTypeMessage.getDoubleDataType().getSize();
			ByteBuffer data = rootObject.readBufferFromAddressNoOrder(dataLayoutMessage.getAddress() + rowPos,
				(int) (rootObject.getCol() * this.dataTypeMessage.getDoubleDataType().getSize()));
			data.order(LITTLE_ENDIAN);

			return data;
		}
		catch(Exception e) {
			throw new H5RuntimeException("Failed to map data buffer for dataset", e);
		}
	}
	public H5DataTypeMessage getDataType() {
		return dataTypeMessage;
	}
}
