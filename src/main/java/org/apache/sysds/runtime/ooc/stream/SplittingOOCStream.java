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

public class SplittingOOCStream<T> implements OOCStream<T> {
	private OOCStream<T> _sourceStream;
	private SubOOCStream<T>[] _subStreams;

	public SplittingOOCStream(OOCStream<T> sourceStream, Function<T, Integer> partitionFunc, int numPartitions) {
		_sourceStream = sourceStream;
		_subStreams = new SubOOCStream[numPartitions];
		for(int i = 0; i < numPartitions; i++)
			_subStreams[i] = new SubOOCStream<>(this);

		_sourceStream.setSubscriber(cb -> {
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

						for(int i = 0; i < numPartitions; i++) {
							SubOOCStream<T> current = _subStreams[i];
							if(current != null)
								current.propagateFailure(failure);
						}
						return;
					}

					if(cb.isEos()) {
						SubOOCStream<T> current;
						for(int i = 0; i < numPartitions; i++) {
							// This requires no additional locking because we know EOS
							// is always triggered after the last non EOS call finished
							current = _subStreams[i];
							if(current != null)
								current.closeInput();
						}
						return;
					}

					if(cb instanceof OOCStream.GroupQueueCallback<?>) {
						@SuppressWarnings("unchecked")
						OOCStream.GroupQueueCallback<T> group = (OOCStream.GroupQueueCallback<T>) cb;
						for(int gi = 0; gi < group.size(); gi++) {
							OOCStream.QueueCallback<T> sub = group.getCallback(gi);
							try(sub) {
								int partition = partitionFunc.apply(sub.get());
								if(partition < 0 || partition >= numPartitions)
									throw new DMLRuntimeException("Invalid partition index: " + partition + " for " + numPartitions + " partitions");
								_subStreams[partition].enqueue(sub.keepOpen());
							}
						}
						return;
					}

					int partition = partitionFunc.apply(cb.get());
					if(partition < 0 || partition >= numPartitions)
						throw new DMLRuntimeException("Invalid partition index: " + partition + " for " + numPartitions + " partitions");
					_subStreams[partition].enqueue(cb.keepOpen());
				}
			}
			catch(DMLRuntimeException re) {
				propagateFailure(re);
			}
		});
	}

	public OOCStream<T> getSubStream(int idx) {
		return _subStreams[idx];
	}

	@Override
	public void enqueue(T t) {
		throw new UnsupportedOperationException();
	}

	@Override
	public T dequeue() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void closeInput() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void propagateFailure(DMLRuntimeException re) {
		_sourceStream.propagateFailure(re);
		for(SubOOCStream<T> subStream : _subStreams)
			subStream.propagateFailure(re);
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
		throw new UnsupportedOperationException();
	}

	@Override
	public OOCStream<T> getReadStream() {
		throw new UnsupportedOperationException();
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
		return _sourceStream.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _sourceStream.getData();
	}

	@Override
	public void setData(CacheableData<?> data) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		_sourceStream.messageUpstream(msg);
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		for(SubOOCStream<T> sub : _subStreams)
			sub.messageDownstream(msg);
	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void clearUpstreamMessageRelays() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void clearDownstreamMessageRelays() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		throw new UnsupportedOperationException();
	}
}
