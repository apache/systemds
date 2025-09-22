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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

import java.util.ArrayList;

/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 *
 */
public class ResettableStream extends LocalTaskQueue<IndexedMatrixValue> {

	// original live stream
	private final LocalTaskQueue<IndexedMatrixValue> _source;

	// in-memory cache to store stream for re-play
	private final ArrayList<IndexedMatrixValue> _cache;

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private int _replayPosition = 0; // slider position in the stream

	public ResettableStream(LocalTaskQueue<IndexedMatrixValue> source) {
		_source = source;
		_cache = new ArrayList<>();
	}

	/**
	 * Dequeues a task. If it is the first, it reads from the disk and stores in the cache.
	 * For subsequent passes it reads from the memory.
	 *
	 * @return The next matrix value in the stream, or NO_MORE_TASKS
	 * @throws InterruptedException
	 */
	@Override
	public synchronized IndexedMatrixValue dequeueTask()
					throws InterruptedException {
		if (_cacheInProgress) {
			// First pass: Read value from the source and cache it, and return.
			IndexedMatrixValue task = _source.dequeueTask();
			if (task != NO_MORE_TASKS) {
				_cache.add(new IndexedMatrixValue(task));
			} else {
				_cacheInProgress = false; // caching is complete
				_source.closeInput(); // close source stream
			}
			notifyAll(); // Notify all the waiting consumers waiting for cache to fill with this stream
			return task;
		} else {
			// Replay pass: read directly from in-memory cache
			if (_replayPosition < _cache.size()) {
				// Return a copy to ensure consumer won't modify the cache
				return new IndexedMatrixValue(_cache.get(_replayPosition++));
			} else {
				return (IndexedMatrixValue) NO_MORE_TASKS;
			}
		}
	}

	/**
	 * Resets the stream to beginning to read the stream from start.
	 * This can only be called once the stream is fully consumed once.
	 */
	public synchronized void reset() throws InterruptedException {
		if (_cacheInProgress) {
			// Attempted to reset a stream that's not been fully cached yet.
			wait();
		}
		_replayPosition = 0;
	}

	@Override
	public synchronized void closeInput() {
		_source.closeInput();
	}
}
