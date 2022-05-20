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
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public class Memorizer {
	private final AComEst _sEst;
	private final Map<ColIndexes, CompressedSizeInfoColGroup> mem;
	private int st1 = 0, st2 = 0, st3 = 0, st4 = 0;

	public Memorizer(AComEst sEst) {
		_sEst = sEst;
		mem = new HashMap<>();
	}

	public void put(CompressedSizeInfoColGroup g) {
		mem.put(new ColIndexes(g.getColumns()), g);
	}

	public void put(ColIndexes key, CompressedSizeInfoColGroup val) {
		mem.put(key, val);
	}

	public CompressedSizeInfoColGroup get(ColIndexes c) {
		return mem.get(c);
	}

	public void remove(ColIndexes c1, ColIndexes c2) {
		mem.remove(c1);
		mem.remove(c2);
		Iterator<Entry<ColIndexes, CompressedSizeInfoColGroup>> i = mem.entrySet().iterator();
		while(i.hasNext()) {
			final ColIndexes eci = i.next().getKey();
			if(eci.contains(c1, c2))
				i.remove();
		}
	}

	public CompressedSizeInfoColGroup getOrCreate(ColIndexes cI, ColIndexes c1, ColIndexes c2){
		CompressedSizeInfoColGroup g = mem.get(cI);
		st2++;
		if(g == null) {
			final CompressedSizeInfoColGroup left = mem.get(c1);
			final CompressedSizeInfoColGroup right = mem.get(c2);
			if(left != null && right != null) {
				st3++;
				g = _sEst.combine(cI._indexes, left, right);

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

	public void incst4() {
		st4++;
	}

	public String stats() {
		return " possible: " + st1 + " requests: " + st2 + " combined: " + st3  + " outSecond: "+ st4;
	}

	public void resetStats() {
		st1 = 0;
		st2 = 0;
		st3 = 0;
		st4 = 0;
	}

	@Override
	public String toString() {
		return mem.toString();
	}
}
