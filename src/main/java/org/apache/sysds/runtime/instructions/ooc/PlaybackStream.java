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

public class PlaybackStream implements OOCStream<IndexedMatrixValue>, OOCStreamable<IndexedMatrixValue> {
	private final CachingStream _streamCache;
	private int _streamIdx;

	public PlaybackStream(CachingStream streamCache) {
		this._streamCache = streamCache;
		this._streamIdx = 0;
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
		final SubscribableTaskQueue<IndexedMatrixValue> q = new SubscribableTaskQueue<>();
		setSubscriber(() -> q.enqueue(dequeue()));
		return q;
	}

	@Override
	public synchronized IndexedMatrixValue dequeue() {
		try {
			return _streamCache.get(_streamIdx++);
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
	public void setSubscriber(Runnable subscriber) {
		_streamCache.setSubscriber(subscriber);
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
