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

import org.apache.sysds.runtime.controlprogram.caching.OOCEvictionManager;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;


/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 * <p>
 * Uses OOCEvictionManager for out-of-core caching.
 *
 */
public class ResettableStream extends LocalTaskQueue<IndexedMatrixValue> {

	// original live stream
	private final LocalTaskQueue<IndexedMatrixValue> _source;

	private static final IDSequence _streamSeq = new  IDSequence();
	// stream identifier
	private final long _streamId;

	// block counter
	private int _numBlocks = 0;


	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private int _replayPosition = 0; // slider position in the stream

	public ResettableStream(LocalTaskQueue<IndexedMatrixValue> source) {
		this(source, _streamSeq.getNextID());
	}
	public ResettableStream(LocalTaskQueue<IndexedMatrixValue> source, long streamId) {
		_source = source;
		_streamId = streamId;
	}

	/**
	 * Dequeues a task. If it is the first, it reads from the disk and stores in the cache.
	 * For subsequent passes it reads from the memory.
	 *
	 * @return The next matrix value in the stream, or NO_MORE_TASKS
	 */
	@Override
	public synchronized IndexedMatrixValue dequeueTask()
					throws InterruptedException {
		if (_cacheInProgress) {
			// First pass: Read value from the source and cache it, and return.
			IndexedMatrixValue task = _source.dequeueTask();
			if (task != NO_MORE_TASKS) {

				OOCEvictionManager.put(_streamId, _numBlocks, task);
				_numBlocks++;

				return task;
			} else {
				_cacheInProgress = false; // caching is complete
				_source.closeInput(); // close source stream

				// Notify all the waiting consumers waiting for cache to fill with this stream
				notifyAll();
				return (IndexedMatrixValue) NO_MORE_TASKS;
			}
		} else {
			// Replay pass: read from the buffer
			if (_replayPosition < _numBlocks) {
				return OOCEvictionManager.get(_streamId, _replayPosition++);
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
		while (_cacheInProgress) {
			// Attempted to reset a stream that's not been fully cached yet.
			wait();
		}
		_replayPosition = 0;
	}

	@Override
	public synchronized void closeInput() {
		_source.closeInput();
	}
	
	@Override
	public synchronized boolean isProcessed() {
		return false;
	}
}
