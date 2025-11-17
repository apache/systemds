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

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class PlaybackStream implements OOCStream<IndexedMatrixValue>, OOCStreamable<IndexedMatrixValue> {
	private final CachingStream _streamCache;
	private final AtomicInteger _streamIdx;
	private final AtomicInteger _taskCtr;
	private final AtomicBoolean _subscriberSet;

	public PlaybackStream(CachingStream streamCache) {
		this._streamCache = streamCache;
		this._streamIdx = new AtomicInteger(0);
		this._taskCtr = new AtomicInteger(1);
		this._subscriberSet = new AtomicBoolean(false);
		streamCache.incrSubscriberCount(1);
	}

	@Override
	public void enqueue(IndexedMatrixValue t) {
		throw new DMLRuntimeException("Cannot enqueue to a playback stream");
	}

	@Override
	public void closeInput() {
		throw new DMLRuntimeException("Cannot close a playback stream");
	}

	@Override
	public LocalTaskQueue<IndexedMatrixValue> toLocalTaskQueue() {
		final LocalTaskQueue<IndexedMatrixValue> q = new LocalTaskQueue<>();
		setSubscriber(val -> {
			if (val.get() == null) {
				q.closeInput();
				return;
			}
			try {
				q.enqueueTask(val.get());
			}
			catch(InterruptedException e) {
				throw new RuntimeException(e);
			}
		});
		return q;
	}

	@Override
	public IndexedMatrixValue dequeue() {
		if (_subscriberSet.get())
			throw new IllegalStateException("Cannot dequeue from a playback stream if a subscriber has been set");

		try {
			return _streamCache.get(_streamIdx.getAndIncrement());
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReadStream() {
		return _streamCache.getReadStream();
	}

	@Override
	public OOCStream<IndexedMatrixValue> getWriteStream() {
		return _streamCache.getWriteStream();
	}

	@Override
	public boolean isProcessed() {
		return false;
	}

	@Override
	public void setSubscriber(Consumer<QueueCallback<IndexedMatrixValue>> subscriber) {
		if (!_subscriberSet.compareAndSet(false, true))
			throw new IllegalArgumentException("Subscriber cannot be set multiple times");

		/**
		 * To guarantee that NO_MORE_TASKS is invoked after all subscriber calls
		 * finished, we keep track of running tasks using a task counter.
		 */
		_streamCache.setSubscriber(() -> {
			try {
				_taskCtr.incrementAndGet();

				IndexedMatrixValue val;

				try {
					val = _streamCache.get(_streamIdx.getAndIncrement());
				} catch (InterruptedException e) {
					throw new DMLRuntimeException(e);
				}

				if (val != null)
					subscriber.accept(new QueueCallback<>(val, null));

				if (_taskCtr.addAndGet(val == null ? -2 : -1) == 0)
					subscriber.accept(new QueueCallback<>(null, null));
			} catch (DMLRuntimeException e) {
				subscriber.accept(new QueueCallback<>(null, e));
			}
		}, false);
	}

	@Override
	public void propagateFailure(DMLRuntimeException re) {
		_streamCache.getWriteStream().propagateFailure(re);
	}

	@Override
	public boolean hasStreamCache() {
		return true;
	}

	@Override
	public CachingStream getStreamCache() {
		return _streamCache;
	}
}
