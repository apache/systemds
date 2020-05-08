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

package org.apache.sysds.runtime.util;

/**
 * Inspired by:
 * https://stackoverflow.com/questions/15736626/java-how-to-create-and-manipulate-a-bit-array-with-length-of-10-million-bits
 * 
 */
public class BitArray {
	private final long[] _values;

	public BitArray(long length) {
		_values = new long[(int) (length / 64L) + (length % 64L == 0 ? 0 : 1)];
	}

	public long getLength() {
		return ((long) _values.length) * 64L;
	}

	public boolean get(long index) {
		return (_values[(int) (index / 64L)] & (1L << (index % 64L))) != 0L;
	}

	public void set(long index, boolean value) {
		long chunk = _values[(int) (index / 64L)];
		long bitIndex = 1L << (index % 64L);
		if(value) {
			chunk |= bitIndex;
		}
		else {
			chunk &= (0xFFFFFFFFFFFFFFFFL - bitIndex);
		}
		_values[(int) (index / 64L)] = chunk;
	}

	public long getChunk(int index) {
		return _values[index];
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		for(int x = _values.length - 1; x >= 0; x--) {
			sb.append(String.format("%016x", _values[x]));
		}
		return sb.reverse().toString();
	}
}