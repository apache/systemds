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

package org.apache.sysds.runtime.ooc.stream;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.function.BiFunction;
import java.util.function.Consumer;

public class SubOOCStream<T> implements OOCStream<T> {
	private OOCStream<T> _sourceStream;
	private SubscribableTaskQueue<QueueCallback<T>> _taskQueue;
	private QueueCallback<T> _last;

	public SubOOCStream(OOCStream<T> sourceStream) {
		_sourceStream = sourceStream;
		_taskQueue = new SubscribableTaskQueue<>();
		_taskQueue.setUpstreamMessageRelay(_sourceStream::messageUpstream);
	}

	public void enqueue(QueueCallback<T> callback) {
		_taskQueue.enqueue(callback);
	}

	@Override
	public void enqueue(T t) {
		_taskQueue.enqueue(new SimpleQueueCallback<>(t, null));
	}

	@Override
	public synchronized T dequeue() {
		if(_last != null)
			_last.close();
		_last = _taskQueue.dequeue();
		if(_last != null)
			return _last.get();
		return null;
	}

	@Override
	public void closeInput() {
		_taskQueue.closeInput();
	}

	@Override
	public void propagateFailure(DMLRuntimeException re) {
		_taskQueue.propagateFailure(re);
	}

	@Override
	public boolean hasStreamCache() {
		return _sourceStream.hasStreamCache();
	}

	@Override
	public CachingStream getStreamCache() {
		return _sourceStream.getStreamCache();
	}

	@Override
	public void setSubscriber(Consumer<QueueCallback<T>> subscriber) {
		_taskQueue.setSubscriber(cb -> {
			if(cb.isEos()) {
				subscriber.accept(OOCStream.eos(null));
				return;
			}
			if(cb.isFailure()) {
				try {
					cb.get();
					subscriber.accept(OOCStream.eos(new DMLRuntimeException("Stream callback indicated failure without cause")));
				}
				catch(DMLRuntimeException re) {
					subscriber.accept(OOCStream.eos(re));
				}
			}
			else
				subscriber.accept(cb.get());
		});
	}

	@Override
	public OOCStream<T> getReadStream() {
		return this;
	}

	@Override
	public OOCStream<T> getWriteStream() {
		return _sourceStream.getWriteStream();
	}

	@Override
	public boolean isProcessed() {
		return _sourceStream.isProcessed();
	}

	@Override
	public DataCharacteristics getDataCharacteristics() {
		return _taskQueue.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _taskQueue.getData();
	}

	@Override
	public void setData(CacheableData<?> data) {
		_taskQueue.setData(data);
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		_taskQueue.messageUpstream(msg);
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		_taskQueue.messageDownstream(msg);
	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		// Upstream is handled by source stream
		throw new UnsupportedOperationException();
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_taskQueue.setDownstreamMessageRelay(relay);
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_taskQueue.addDownstreamMessageRelay(relay);
	}

	@Override
	public void clearUpstreamMessageRelays() {
		_taskQueue.clearUpstreamMessageRelays();
	}

	@Override
	public void clearDownstreamMessageRelays() {
		_taskQueue.clearDownstreamMessageRelays();
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		_taskQueue.setIXTransform(transform);
	}
}
