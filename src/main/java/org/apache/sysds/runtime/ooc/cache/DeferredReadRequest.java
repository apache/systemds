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

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

class DeferredReadRequest {
	static final short NOT_SCHEDULED = 0;
	static final short SCHEDULED = 1;
	static final short PINNED = 2;

	private final CompletableFuture<List<BlockEntry>> _future;
	private final List<BlockEntry> _entries;
	private final short[] _pinned;
	private final boolean[] _retainHinted;
	private final AtomicInteger _availableCount;
	private double _priorityScore;
	private long _sequence;

	DeferredReadRequest(CompletableFuture<List<BlockEntry>> future, List<BlockEntry> entries) {
		this._future = future;
		this._entries = entries;
		this._pinned = new short[entries.size()];
		this._retainHinted = new boolean[entries.size()];
		this._availableCount = new AtomicInteger(0);
		this._priorityScore = 0;
		this._sequence = 0;
	}

	CompletableFuture<List<BlockEntry>> getFuture() {
		return _future;
	}

	List<BlockEntry> getEntries() {
		return _entries;
	}

	synchronized void setPriorityScore(double score) {
		_priorityScore = score;
	}

	synchronized void addPriorityScore(double delta) {
		_priorityScore += delta;
	}

	synchronized double getPriorityScore() {
		return _priorityScore;
	}

	synchronized void setSequence(long sequence) {
		_sequence = sequence;
	}

	synchronized long getSequence() {
		return _sequence;
	}

	synchronized boolean actionRequired(int idx) {
		return _pinned[idx] == NOT_SCHEDULED;
	}

	synchronized boolean setPinned(int idx) {
		if(_pinned[idx] == PINNED)
			return false; // already pinned
		_pinned[idx] = PINNED;
		return _availableCount.incrementAndGet() == _entries.size();
	}

	synchronized void schedule(int idx) {
		_pinned[idx] = SCHEDULED;
	}

	synchronized void markRetainHinted(int idx) {
		_retainHinted[idx] = true;
	}

	synchronized boolean isRetainHinted(int idx) {
		return _retainHinted[idx];
	}

	boolean isComplete() {
		return _availableCount.get() == _entries.size();
	}
}
