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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * This class abstracts over an encrypted data. It stores the data as opaque byte array. The layout is unspecified.
 */
public abstract class Encrypted extends Data {
	private static final long serialVersionUID = 1762936872268046168L;

	private final int[] _dims;
	private final DataCharacteristics _dc;
	private final byte[] _data;

	public Encrypted(int[] dims, DataCharacteristics dc, byte[] data, Types.DataType dt) {
		super(dt, Types.ValueType.UNKNOWN);
		_dims = dims;
		_dc = dc;
		_data = data;
	}

	public int[] getDims() {
		return _dims;
	}

	public DataCharacteristics getDataCharacteristics() {
		return _dc;
	}

	public byte[] getData() {
		return _data;
	}
}
