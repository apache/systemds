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
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public class PlaybackStream implements OOCStream<IndexedMatrixValue> {
	private final CachingStream _streamCache;
	private final AtomicInteger _streamIdx;
	private final AtomicBoolean _subscriberSet;
	private QueueCallback<IndexedMatrixValue> _lastDequeue;
	private volatile CopyOnWriteArrayList<Consumer<OOCStreamMessage>> _downstreamRelays;

	public PlaybackStream(CachingStream streamCache) {
		this._streamCache = streamCache;
		this._streamIdx = new AtomicInteger(0);
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
	public synchronized IndexedMatrixValue dequeue() {
		if (_subscriberSet.get())
			throw new IllegalStateException("Cannot dequeue from a playback stream if a subscriber has been set");

		try {
			if (_lastDequeue != null)
				_lastDequeue.close();
			_lastDequeue = _streamCache.get(_streamIdx.getAndIncrement()).get();
			return _lastDequeue.get();
		} catch (InterruptedException | ExecutionException e) {
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
	public DataCharacteristics getDataCharacteristics() {
		return _streamCache.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _streamCache.getData();
	}

	@Override
	public void setData(CacheableData<?> data) {
		_streamCache.setData(data);
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		if(msg.isCancelled())
			return;
		_streamCache.messageUpstream(msg);
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		if(msg.isCancelled())
			return;
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamRelays;
		if (relays != null) {
			for (Consumer<OOCStreamMessage> relay : relays) {
				if (msg.isCancelled())
					break;
				relay.accept(msg);
			}
		}
	}

	@Override
	public void setSubscriber(Consumer<QueueCallback<IndexedMatrixValue>> subscriber) {
		if (!_subscriberSet.compareAndSet(false, true))
			throw new IllegalArgumentException("Subscriber cannot be set multiple times");

		_streamCache.setSubscriber(subscriber, false);
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

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		addDownstreamMessageRelay(relay);
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		if (relay == null)
			throw new IllegalArgumentException("Cannot set downstream relay to null");
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamRelays;
		if (relays == null) {
			synchronized(this) {
				if (_downstreamRelays == null)
					_downstreamRelays = new CopyOnWriteArrayList<>();
				relays = _downstreamRelays;
			}
		}
		relays.add(0, relay);
		_streamCache.addDownstreamMessageRelay(relay);
	}

	@Override
	public void clearUpstreamMessageRelays() {
		// No upstream relays supported
	}

	@Override
	public void clearDownstreamMessageRelays() {
		_downstreamRelays = null;
		_streamCache.clearDownstreamMessageRelays();
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		throw new UnsupportedOperationException();
	}
}
