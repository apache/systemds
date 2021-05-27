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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class FillValueOldMessage extends Message {

	private final ByteBuffer fillValue;

	FillValueOldMessage(ByteBuffer bb, BitSet flags) {
		super(flags);

		final int size = Utils.readBytesAsUnsignedInt(bb, 4);
		fillValue = Utils.createSubBuffer(bb, size);
	}

	/**
	 * The fill value. The bytes of the fill value are interpreted using the same
	 * datatype as for the dataset.
	 *
	 * @return a buffer containing the fill value
	 */
	public ByteBuffer getFillValue() {
		return fillValue.asReadOnlyBuffer();
	}

}
