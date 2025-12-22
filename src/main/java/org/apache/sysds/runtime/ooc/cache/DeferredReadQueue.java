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

package org.apache.sysds.runtime.ooc.cache;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

class DeferredReadQueue {
	private final ArrayList<DeferredReadRequest> _heap;
	private final Map<DeferredReadRequest, Integer> _index;
	private final Map<BlockKey, Set<DeferredReadRequest>> _byKey;
	private long _seqCounter;

	DeferredReadQueue() {
		this._heap = new ArrayList<>();
		this._index = new IdentityHashMap<>();
		this._byKey = new HashMap<>();
		this._seqCounter = 0;
	}

	boolean isEmpty() {
		return _heap.isEmpty();
	}

	int size() {
		return _heap.size();
	}

	DeferredReadRequest peek() {
		return _heap.isEmpty() ? null : _heap.get(0);
	}

	DeferredReadRequest poll() {
		if(_heap.isEmpty())
			return null;
		DeferredReadRequest req = _heap.get(0);
		removeAt(0);
		removeFromIndex(req);
		return req;
	}

	void add(DeferredReadRequest req) {
		req.setSequence(_seqCounter++);
		_heap.add(req);
		_index.put(req, _heap.size() - 1);
		addToIndex(req);
		heapifyUp(_heap.size() - 1);
	}

	void remove(DeferredReadRequest req) {
		Integer idx = _index.get(req);
		if(idx == null)
			return;
		removeAt(idx);
		removeFromIndex(req);
	}

	void clear() {
		_heap.clear();
		_index.clear();
		_byKey.clear();
		_seqCounter = 0;
	}

	boolean boost(BlockKey key, double priority) {
		if(priority == 0)
			return false;
		Set<DeferredReadRequest> requests = _byKey.get(key);
		if(requests == null || requests.isEmpty())
			return false;
		for(DeferredReadRequest req : requests) {
			double delta = priority / req.getEntries().size();
			req.addPriorityScore(delta);
			updatePosition(req);
		}
		return true;
	}

	private void updatePosition(DeferredReadRequest req) {
		Integer idx = _index.get(req);
		if(idx == null)
			return;
		int parent = (idx - 1) / 2;
		if(idx > 0 && compare(req, _heap.get(parent)) > 0)
			heapifyUp(idx);
		else
			heapifyDown(idx);
	}

	private void addToIndex(DeferredReadRequest req) {
		for(BlockEntry entry : req.getEntries()) {
			BlockKey key = entry.getKey();
			Set<DeferredReadRequest> set = _byKey.get(key);
			if(set == null) {
				set = Collections.newSetFromMap(new IdentityHashMap<>());
				_byKey.put(key, set);
			}
			set.add(req);
		}
	}

	private void removeFromIndex(DeferredReadRequest req) {
		for(BlockEntry entry : req.getEntries()) {
			BlockKey key = entry.getKey();
			Set<DeferredReadRequest> set = _byKey.get(key);
			if(set == null)
				continue;
			set.remove(req);
			if(set.isEmpty())
				_byKey.remove(key);
		}
	}

	private DeferredReadRequest removeAt(int idx) {
		int lastIdx = _heap.size() - 1;
		DeferredReadRequest removed = _heap.get(idx);
		DeferredReadRequest last = _heap.get(lastIdx);
		_heap.set(idx, last);
		_heap.remove(lastIdx);
		_index.remove(removed);
		if(idx < _heap.size()) {
			_index.put(last, idx);
			updatePosition(last);
		}
		return removed;
	}

	private void heapifyUp(int idx) {
		int i = idx;
		while(i > 0) {
			int parent = (i - 1) / 2;
			if(compare(_heap.get(i), _heap.get(parent)) <= 0)
				break;
			swap(i, parent);
			i = parent;
		}
	}

	private void heapifyDown(int idx) {
		int i = idx;
		int size = _heap.size();
		while(true) {
			int left = i * 2 + 1;
			if(left >= size)
				break;
			int right = left + 1;
			int best = left;
			if(right < size && compare(_heap.get(right), _heap.get(left)) > 0)
				best = right;
			if(compare(_heap.get(best), _heap.get(i)) <= 0)
				break;
			swap(i, best);
			i = best;
		}
	}

	private void swap(int i, int j) {
		DeferredReadRequest tmp = _heap.get(i);
		_heap.set(i, _heap.get(j));
		_heap.set(j, tmp);
		_index.put(_heap.get(i), i);
		_index.put(_heap.get(j), j);
	}

	private int compare(DeferredReadRequest a, DeferredReadRequest b) {
		int byPriority = Double.compare(a.getPriorityScore(), b.getPriorityScore());
		if(byPriority != 0)
			return byPriority;
		return Long.compare(b.getSequence(), a.getSequence());
	}
}
