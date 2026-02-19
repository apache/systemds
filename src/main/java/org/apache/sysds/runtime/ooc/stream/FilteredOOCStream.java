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
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class FilteredOOCStream<T> implements OOCStream<T> {
	private final OOCStream<T> _sourceStream;
	private final Function<T, Boolean> _predicate;
	private CacheableData<?> _data;

	public FilteredOOCStream(OOCStream<T> sourceStream, Function<T, Boolean> predicate) {
		_sourceStream = sourceStream;
		_predicate = predicate;
	}

	@Override
	public void enqueue(T t) {
		_sourceStream.enqueue(t);
	}

	@Override
	public synchronized T dequeue() {
		T next;
		while((next = _sourceStream.dequeue()) != null) {
			if(_predicate.apply(next))
				return next;
		}
		return null;
	}

	@Override
	public void closeInput() {
		_sourceStream.closeInput();
	}

	@Override
	public void propagateFailure(DMLRuntimeException re) {
		_sourceStream.propagateFailure(re);
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
		_sourceStream.setSubscriber(cb -> {
			if(cb.isFailure() || cb.isEos() || _predicate.apply(cb.get()))
				subscriber.accept(cb);
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
		return _data == null ? null : _data.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _data;
	}

	@Override
	public void setData(CacheableData<?> data) {
		_data = data;
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		_sourceStream.messageUpstream(msg);
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		_sourceStream.messageDownstream(msg);
	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_sourceStream.setUpstreamMessageRelay(relay);
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_sourceStream.setDownstreamMessageRelay(relay);
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_sourceStream.addUpstreamMessageRelay(relay);
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		_sourceStream.addDownstreamMessageRelay(relay);
	}

	@Override
	public void clearUpstreamMessageRelays() {
		_sourceStream.clearUpstreamMessageRelays();
	}

	@Override
	public void clearDownstreamMessageRelays() {
		_sourceStream.clearDownstreamMessageRelays();
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		_sourceStream.setIXTransform(transform);
	}
}
