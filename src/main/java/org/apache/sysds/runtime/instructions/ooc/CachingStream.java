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
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

import java.util.HashMap;
import java.util.Map;

/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 *
 */
public class CachingStream implements OOCStreamable<IndexedMatrixValue> {

	public static final IDSequence _streamSeq = new IDSequence();

	// original live stream
	private final OOCStream<IndexedMatrixValue> _source;

	// stream identifier
	private final long _streamId;

	// block counter
	private int _numBlocks = 0;

	private Runnable[] _subscribers;

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private Map<MatrixIndexes, Integer> _index;

	private DMLRuntimeException _failure;

	public CachingStream(OOCStream<IndexedMatrixValue> source) {
		this(source, _streamSeq.getNextID());
	}

	public CachingStream(OOCStream<IndexedMatrixValue> source, long streamId) {
		_source = source;
		_streamId = streamId;
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
			} catch (DMLRuntimeException e) {
				// Propagate failure to subscribers
				_failure = e;
				synchronized (this) {
					notifyAll();
				}

				Runnable[] mSubscribers = _subscribers;
				if(mSubscribers != null) {
					for(Runnable mSubscriber : mSubscribers) {
						try {
							mSubscriber.run();
						} catch (Exception ignored) {
						}
					}
				}
			}
		});
	}

	private synchronized boolean fetchFromStream() throws InterruptedException {
		if(!_cacheInProgress)
			throw new DMLRuntimeException("Stream is closed");

		IndexedMatrixValue task = _source.dequeue();

		if(task != LocalTaskQueue.NO_MORE_TASKS) {
			OOCEvictionManager.put(_streamId, _numBlocks, task);
			if (_index != null)
				_index.put(task.getIndexes(), _numBlocks);
			_numBlocks++;
			notifyAll();
			return false;
		}
		else {
			_cacheInProgress = false; // caching is complete
			notifyAll();
			return true;
		}
	}

	public synchronized IndexedMatrixValue get(int idx) throws InterruptedException {
		while (true) {
			if (_failure != null)
				throw _failure;
			else if (idx < _numBlocks) {
				IndexedMatrixValue out = OOCEvictionManager.get(_streamId, idx);

				if (_index != null) // Ensure index is up to date
					_index.putIfAbsent(out.getIndexes(), idx);

				return out;
			} else if (!_cacheInProgress)
				return (IndexedMatrixValue)LocalTaskQueue.NO_MORE_TASKS;

			wait();
		}
	}

	public synchronized IndexedMatrixValue findCached(MatrixIndexes idx) {
		return OOCEvictionManager.get(_streamId, _index.get(idx));
	}

	public synchronized void activateIndexing() {
		if (_index == null)
			_index = new HashMap<>();
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReadStream() {
		return new PlaybackStream(this);
	}

	@Override
	public OOCStream<IndexedMatrixValue> getWriteStream() {
		return _source.getWriteStream();
	}

	@Override
	public boolean isProcessed() {
		return false;
	}

	@Override
	public void setSubscriber(Runnable subscriber) {
		int mNumBlocks;
		synchronized (this) {
			mNumBlocks = _numBlocks;
			if (_cacheInProgress) {
				int newLen = _subscribers == null ? 1 : _subscribers.length + 1;
				Runnable[] newSubscribers = new Runnable[newLen];

				if(newLen > 1)
					System.arraycopy(_subscribers, 0, newSubscribers, 0, newLen - 1);

				newSubscribers[newLen - 1] = subscriber;
				_subscribers = newSubscribers;
			}
		}

		for (int i = 0; i < mNumBlocks; i++)
			subscriber.run();

		if (!_cacheInProgress)
			subscriber.run(); // To fetch the NO_MORE_TASK element
	}
}
