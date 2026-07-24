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

package org.apache.sysds.runtime.ooc.cache.eviction;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockState;
import org.apache.sysds.runtime.ooc.cache.collections.MaskedOnceArrayList;

import java.util.PriorityQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.LongUnaryOperator;

public class EvictController {
	private final CopyOnWriteArrayList<LongUnaryOperator> _op = new CopyOnWriteArrayList<>();

	public void addEvictionPolicy(LongUnaryOperator op) {
		if(op == null)
			throw new IllegalArgumentException("Eviction policy must not be null.");
		_op.add(op);
	}

	public void findEvictionCandidates(MaskedOnceArrayList<BlockEntry> list,
		PriorityQueue<IndexedObjectPair<BlockEntry>> candidates, int k, long estimatedReuseTimestamp) {
		if(_op.isEmpty()) {
			list.forEachLive((idx, b) -> {
				if(!isEvictionCandidate(b))
					return true;
				var iop = new IndexedObjectPair<>(estimatedReuseTimestamp + idx, b);
				if(candidates.size() < k) {
					candidates.offer(iop);
				}
				else if(iop.compareTo(candidates.peek()) > 0) {
					candidates.poll();
					candidates.offer(iop);
				}
				return true;
			}, true);
			return;
		}
		list.forEachLive((idx, b) -> {
			if(!isEvictionCandidate(b))
				return true;
			long score = computeScore(idx);
			if(score == Long.MAX_VALUE)
				score = idx + estimatedReuseTimestamp;
			var iop = new IndexedObjectPair<>(score, b);
			if(candidates.size() < k) {
				candidates.offer(iop);
			}
			else if(iop.compareTo(candidates.peek()) > 0) {
				candidates.poll();
				candidates.offer(iop);
			}
			return true;
		}, true);
	}

	private boolean isEvictionCandidate(BlockEntry entry) {
		BlockState state = entry.getState();
		return state == BlockState.HOT || state == BlockState.WARM;
	}

	private long computeScore(int idx) {
		long out = Long.MAX_VALUE;
		for(LongUnaryOperator uop : _op)
			out = Math.min(out, uop.applyAsLong(idx));
		return out;
	}
}
