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

package org.apache.sysds.runtime.ooc.cache.io;

import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public final class SpillableObjectRegistry {
	private static final byte INDEXED_MATRIX_VALUE = 1;

	private SpillableObjectRegistry() {
	}

	public static boolean tryWrite(DataOutput out, SpillableObject obj) throws IOException {
		byte type = typeOf(obj);
		out.writeByte(type);
		return obj.tryWrite(out);
	}

	public static SpillableObject read(DataInput in) throws IOException {
		byte type = in.readByte();
		SpillableObject obj = switch(type) {
			case INDEXED_MATRIX_VALUE -> new IndexedMatrixValue();
			default -> throw new IOException("Unknown spillable object type: " + type);
		};
		obj.read(in);
		return obj;
	}

	private static byte typeOf(SpillableObject obj) throws IOException {
		if(obj instanceof IndexedMatrixValue)
			return INDEXED_MATRIX_VALUE;
		throw new IOException("Unsupported spillable object type: " + obj.getClass().getName());
	}
}
