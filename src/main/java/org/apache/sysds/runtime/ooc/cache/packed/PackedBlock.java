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

package org.apache.sysds.runtime.ooc.cache.packed;

import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObjectRegistry;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public final class PackedBlock implements SpillableObject {
	Object[] values;
	long[] sizes;
	long totalSize;

	public PackedBlock() {
		values = null;
		sizes = null;
		totalSize = 0;
	}

	PackedBlock(Object[] values, long[] sizes, long totalSize) {
		this.values = values;
		this.sizes = sizes;
		this.totalSize = totalSize;
	}

	@Override
	public boolean tryWrite(DataOutput out) throws IOException {
		out.writeInt(values.length);
		for(int i = 0; i < values.length; i++) {
			out.writeLong(sizes[i]);
			Object value = values[i];
			if(!(value instanceof SpillableObject spillable))
				return false;
			if(!SpillableObjectRegistry.tryWrite(out, spillable))
				return false;
		}
		return true;
	}

	@Override
	public void read(DataInput in) throws IOException {
		int count = in.readInt();
		values = new Object[count];
		sizes = new long[count];
		totalSize = 0;
		for(int i = 0; i < count; i++) {
			sizes[i] = in.readLong();
			values[i] = SpillableObjectRegistry.read(in);
			totalSize += sizes[i];
		}
	}
}
