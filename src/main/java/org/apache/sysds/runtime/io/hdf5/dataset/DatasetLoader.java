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
import org.apache.sysds.runtime.io.hdf5.api.Dataset;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.ChunkedDataLayoutMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.CompactDataLayoutMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.ContiguousDataLayoutMessage;

public final class DatasetLoader {

	private DatasetLoader() {
		throw new AssertionError("No instances of DatasetLoader");
	}

	public static Dataset createDataset(HdfFileChannel hdfFc, ObjectHeader oh, String name,
			Group parent) {

		final long address = oh.getAddress();
		try {
			// Determine the type of dataset to make
			final DataLayoutMessage dlm = oh.getMessageOfType(DataLayoutMessage.class);

			if (dlm instanceof CompactDataLayoutMessage) {
				return new CompactDataset(hdfFc, address, name, parent, oh);

			} else if (dlm instanceof ContiguousDataLayoutMessage) {
				return new ContiguousDatasetImpl(hdfFc, address, name, parent, oh);

			} else {
				throw new HdfException("Unrecognized Dataset layout type: " + dlm.getClass().getCanonicalName());
			}

		} catch (Exception e) {
			throw new HdfException("Failed to read dataset '" + name + "' at address '" + address + "'", e);
		}
	}

}
