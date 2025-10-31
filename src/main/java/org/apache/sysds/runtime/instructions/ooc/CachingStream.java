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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

import java.util.ArrayList;
import java.util.List;

/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 *
 */
public class CachingStream extends LocalTaskQueue<IndexedMatrixValue> {

	// original live stream
	private final LocalTaskQueue<IndexedMatrixValue> _source;

	// in-memory cache to store stream for re-play
	private final ArrayList<IndexedMatrixValue> _cache;

	private Runnable[] _subscribers;

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.

	public CachingStream(LocalTaskQueue<IndexedMatrixValue> source) {
		_source = source;
		_cache = new ArrayList<>();
		source.setSubscriber(() -> {
			try {
				boolean closed = fetchFromStream();
				Runnable[] mSubscribers = _subscribers;

				if(mSubscribers != null) {
					for(Runnable mSubscriber : mSubscribers)
						mSubscriber.run();

					if (closed) {
						synchronized (this) {
							_subscribers = null;
						}
					}
				}
			} catch (InterruptedException e) {
				throw new DMLRuntimeException(e);
			}
		});
	}

	private boolean fetchFromStream() throws InterruptedException {
		synchronized (this) {
			if(!_cacheInProgress)
				throw new DMLRuntimeException("Stream is closed");
		}

		IndexedMatrixValue task = _source.dequeueTask();

		synchronized (this) {
			if(task != NO_MORE_TASKS) {
				_cache.add(new IndexedMatrixValue(task));
				notifyAll();
				return false;
			}
			else {
				_cacheInProgress = false; // caching is complete
				notifyAll();
				//_source.closeInput(); // close source stream
				return true;
			}
		}
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
		throw new DMLRuntimeException("CachingStream should not be dequeued");
	}

	public synchronized IndexedMatrixValue get(int idx) throws InterruptedException {
		while (true) {
			if (idx < _cache.size())
				return _cache.get(idx);
			else if (!_cacheInProgress)
				return (IndexedMatrixValue)NO_MORE_TASKS;

			wait();
		}
	}

	@Override
	public synchronized void closeInput() {
		throw new DMLRuntimeException("CachingStream should not be closed");
	}
	
	@Override
	public synchronized boolean isProcessed() {
		return false;
	}

	@Override
	public void setSubscriber(Runnable subscriber) {
		synchronized (this) {
			if (_cacheInProgress) {
				int newLen = _subscribers == null ? 1 : _subscribers.length + 1;
				Runnable[] newSubscribers = new Runnable[newLen];

				if(newLen > 1)
					System.arraycopy(_subscribers, 0, newSubscribers, 0, newLen - 1);

				newSubscribers[newLen - 1] = subscriber;
				_subscribers = newSubscribers;
			}
		}

		for (int i = 0; i < _cache.size(); i++)
			subscriber.run();

		if (!_cacheInProgress)
			subscriber.run(); // To fetch the NO_MORE_TASK element
	}
}
