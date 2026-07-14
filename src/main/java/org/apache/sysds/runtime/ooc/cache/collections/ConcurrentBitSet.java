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

package org.apache.sysds.runtime.ooc.cache.collections;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class ConcurrentBitSet {
	private static final VarHandle LONG_ARR = MethodHandles.arrayElementVarHandle(long[].class);

	private final long[] words;

	public ConcurrentBitSet(int bits) {
		// (bits + 63) >>> 6 = ceil(bits / 64.0)
		this.words = new long[(bits + 63) >>> 6];
	}

	public boolean get(int i) {
		int w = i >>> 6;
		long mask = 1L << (i & 63);
		long word = (long) LONG_ARR.getAcquire(words, w);
		return (word & mask) != 0;
	}

	public boolean set(int i) {
		int w = i >>> 6;
		long mask = 1L << (i & 63);

		long prev = (long) LONG_ARR.getAndBitwiseOrRelease(words, w, mask);
		return (prev & mask) == 0; // true if changed absent -> present
	}

	public boolean clear(int i) {
		int w = i >>> 6;
		long mask = 1L << (i & 63);

		long prev = (long) LONG_ARR.getAndBitwiseAndRelease(words, w, ~mask);
		return (prev & mask) != 0; // true if changed present -> absent
	}

	public long getWord(int wordIndex) {
		return (long) LONG_ARR.getAcquire(words, wordIndex);
	}

	public int length() {
		return words.length;
	}
}
