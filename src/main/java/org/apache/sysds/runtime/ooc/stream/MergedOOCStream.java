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

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public class MergedOOCStream<T> implements OOCStream<T> {
	private final List<OOCStream<T>> _sources;
	private final SubscribableTaskQueue<QueueCallback<T>> _taskQueue;
	private final AtomicInteger _openSources;
	private final AtomicBoolean _failed;
	private final CachingStream _sharedCache;
	private QueueCallback<T> _last;

	public MergedOOCStream(List<OOCStream<T>> sources) {
		if(sources == null || sources.isEmpty())
			throw new IllegalArgumentException("MergedOOCStream requires at least one source stream");
		_sources = sources;
		_taskQueue = new SubscribableTaskQueue<>();
		_openSources = new AtomicInteger(sources.size());
		_failed = new AtomicBoolean(false);
		_sharedCache = findSharedCache(sources);

		_taskQueue.setUpstreamMessageRelay(msg -> {
			for(OOCStream<T> source : _sources)
				source.messageUpstream(msg);
		});

		for(OOCStream<T> source : _sources) {
			source.setSubscriber(cb -> {
				try {
					try(cb) {
						if(cb.isFailure()) {
							DMLRuntimeException failure;
							try {
								cb.get();
								failure = new DMLRuntimeException("Stream callback indicated failure without cause");
							}
							catch(DMLRuntimeException re) {
								failure = re;
							}
							propagateFailure(failure);
							return;
						}

						if(cb.isEos()) {
							if(_failed.get())
								return;
							if(_openSources.decrementAndGet() == 0)
								_taskQueue.closeInput();
							return;
						}

						if(_failed.get())
							return;

						_taskQueue.enqueue(cb.keepOpen());
					}
				}
				catch(DMLRuntimeException re) {
					propagateFailure(re);
				}
			});
		}
	}

	@SafeVarargs
	public MergedOOCStream(OOCStream<T>... sources) {
		this(Arrays.asList(sources));
	}

	private static <T> CachingStream findSharedCache(List<OOCStream<T>> sources) {
		CachingStream shared = null;
		for(OOCStream<T> source : sources) {
			if(!source.hasStreamCache())
				return null;
			CachingStream cache = source.getStreamCache();
			if(cache == null)
				return null;
			if(shared == null)
				shared = cache;
			else if(shared != cache)
				return null;
		}
		return shared;
	}

	@Override
	public void enqueue(T t) {
		throw new UnsupportedOperationException();
	}

	@Override
	public synchronized T dequeue() {
		if(_last != null)
			_last.close();
		_last = _taskQueue.dequeue();
		if(_last == null)
			return null;
		return _last.get();
	}

	@Override
	public void closeInput() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void propagateFailure(DMLRuntimeException re) {
		if(!_failed.compareAndSet(false, true))
			return;
		_taskQueue.propagateFailure(re);
		for(OOCStream<T> source : _sources)
			source.propagateFailure(re);
	}

	@Override
	public boolean hasStreamCache() {
		return _sharedCache != null;
	}

	@Override
	public CachingStream getStreamCache() {
		return _sharedCache;
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
				return;
			}
			subscriber.accept(cb.get());
		});
	}

	@Override
	public OOCStream<T> getReadStream() {
		return this;
	}

	@Override
	public OOCStream<T> getWriteStream() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isProcessed() {
		return _taskQueue.isProcessed();
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
