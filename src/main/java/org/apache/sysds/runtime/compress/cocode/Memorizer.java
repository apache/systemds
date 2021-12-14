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

package org.apache.sysds.runtime.compress.cocode;

import java.util.HashMap;
import java.util.Map;

import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.Util;

public class Memorizer {
	private final CompressedSizeEstimator _sEst;
	private final Map<ColIndexes, CompressedSizeInfoColGroup> mem;
	private int st1 = 0, st2 = 0, st3 = 0;

	public Memorizer(CompressedSizeEstimator sEst) {
		_sEst = sEst;
		mem = new HashMap<>();
	}

	public void put(CompressedSizeInfoColGroup g) {
		mem.put(new ColIndexes(g.getColumns()), g);
	}

	public CompressedSizeInfoColGroup get(ColIndexes c) {
		return mem.get(c);
	}

	public void remove(ColIndexes c1, ColIndexes c2) {
		mem.remove(c1);
		mem.remove(c2);
	}

	public CompressedSizeInfoColGroup getOrCreate(ColIndexes c1, ColIndexes c2) {
		final int[] c = Util.join(c1._indexes, c2._indexes);
		final ColIndexes cI = new ColIndexes(c);
		CompressedSizeInfoColGroup g = mem.get(cI);
		st2++;
		if(g == null) {
			final CompressedSizeInfoColGroup left = mem.get(c1);
			final CompressedSizeInfoColGroup right = mem.get(c2);
			if(left != null && right != null) {

				st3++;
				g = _sEst.estimateJoinCompressedSize(c, left, right);

				synchronized(this) {
					mem.put(cI, g);
				}
			}

		}
		return g;
	}

	public void incst1() {
		st1++;
	}

	public String stats() {
		return " possible: " + st1 + " requests: " + st2 + " joined: " + st3;
	}

	public void resetStats() {
		st1 = 0;
		st2 = 0;
		st3 = 0;
	}

	@Override
	public String toString() {
		return mem.toString();
	}
}
