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

package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class OffsetEmpty extends AOffset {

	private static final long serialVersionUID = -471610497392221790L;

	public OffsetEmpty() {

	}

	@Override
	public AIterator getIterator() {
		return null;
	}

	@Override
	public AOffsetIterator getOffsetIterator() {
		return null;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(OffsetFactory.OFF_TYPE_SPECIALIZATIONS.EMPTY.ordinal());
	}

	public static OffsetEmpty readFields(DataInput in) throws IOException {
		return new OffsetEmpty();
	}

	@Override
	public int getOffsetToFirst() {
		return Integer.MAX_VALUE;
	}

	@Override
	public int getOffsetToLast() {
		return Integer.MAX_VALUE;
	}

	@Override
	public long getInMemorySize() {
		return estimateInMemorySize();
	}

	public static long estimateInMemorySize() {
		return 16; // object header
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1; // Byte identifier for Empty
	}

	@Override
	public int getSize() {
		return 0;
	}

	@Override
	public OffsetSliceInfo slice(int l, int u) {
		return new OffsetSliceInfo(-1, -1, this);
	}

	@Override
	protected AOffset moveIndex(int m) {
		return this;
	}

	@Override
	protected int getLength() {
		return 0;
	}
}
