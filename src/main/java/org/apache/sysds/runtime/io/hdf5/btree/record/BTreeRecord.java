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


package org.apache.sysds.runtime.io.hdf5.btree.record;

import org.apache.sysds.runtime.io.hdf5.dataset.chunked.DatasetInfo;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.nio.ByteBuffer;

public abstract class BTreeRecord {

	@SuppressWarnings("unchecked") // Requires that the b-tree is of the correct type for the record
	public static <T extends BTreeRecord> T readRecord(int type, ByteBuffer buffer, DatasetInfo datasetInfo) {
		switch (type) {
			case 0:
				throw new HdfException("b-tree record type 0. Should only be used for testing");
			case 1:
				return (T) new HugeFractalHeapObjectUnfilteredRecord(buffer);
			case 2:
				throw new UnsupportedHdfException("b-tree record type 2. Currently not supported");
			case 3:
				throw new UnsupportedHdfException("b-tree record type 3. Currently not supported");
			case 4:
				throw new UnsupportedHdfException("b-tree record type 4. Currently not supported");
			case 5:
				return (T) new LinkNameForIndexedGroupRecord(buffer);
			case 6:
				throw new UnsupportedHdfException("b-tree record type 6. Currently not supported");
			case 7:
				throw new UnsupportedHdfException("b-tree record type 7. Currently not supported");
			case 8:
				return (T) new AttributeNameForIndexedAttributesRecord(buffer);
			case 9:
				throw new UnsupportedHdfException("b-tree record type 9. Currently not supported");
			case 10:
				return (T) new NonFilteredDatasetChunks(buffer, datasetInfo);
			case 11:
				return (T) new FilteredDatasetChunks(buffer, datasetInfo);
			default:
				throw new HdfException("Unknown b-tree record type. Type = " + type);
		}
	}

}
