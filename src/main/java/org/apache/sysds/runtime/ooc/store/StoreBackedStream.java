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

package org.apache.sysds.runtime.ooc.store;

import java.util.function.Consumer;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;

public final class StoreBackedStream<T extends SpillableObject> implements OOCStream<T> {
	private final OrderedMaterializedStoreReader<T> _reader;
	private volatile DMLRuntimeException _failure;
	private boolean _subscriberSet;
	private OOCStream.QueueCallback<T> _lastDequeue;
	private boolean _exhausted;
	private CacheableData<?> _data;

	public StoreBackedStream(OrderedMaterializedStoreReader<T> reader) {
		_reader = reader;
	}

	@Override
	public void enqueue(T value) {
		throw new DMLRuntimeException("Cannot enqueue to a store-backed stream");
	}

	@Override
	public void enqueue(QueueCallback<T> callback) {
		throw new DMLRuntimeException("Cannot enqueue to a store-backed stream");
	}

	@Override
	public void closeInput() {
		throw new DMLRuntimeException("Cannot close the input of a store-backed stream");
	}

	@Override
	public synchronized T dequeue() {
		QueueCallback<T> callback = dequeueInternal();
		return callback == null ? null : callback.get();
	}

	@Override
	public synchronized QueueCallback<T> dequeueCB() {
		return dequeueInternal();
	}

	private QueueCallback<T> dequeueInternal() {
		if(_subscriberSet)
			throw new IllegalStateException("Cannot dequeue after setting a subscriber");
		if(_lastDequeue != null) {
			_lastDequeue.close();
			_lastDequeue = null;
		}
		if(_failure != null)
			throw _failure;
		if(_exhausted)
			return null;
		try {
			if(!_reader.hasNext()) {
				_exhausted = true;
				_reader.close();
				return null;
			}
			_lastDequeue = new MaterializedCallback<>(_reader.next());
			return _lastDequeue;
		}
		catch(InterruptedException e) {
			Thread.currentThread().interrupt();
			throw recordFailure(e);
		}
		catch(RuntimeException e) {
			throw recordFailure(e);
		}
	}

	@Override
	public synchronized void setSubscriber(Consumer<QueueCallback<T>> subscriber) {
		if(subscriber == null)
			throw new IllegalArgumentException("Cannot set subscriber to null");
		if(_subscriberSet)
			throw new IllegalStateException("Subscriber cannot be set multiple times");
		_subscriberSet = true;
		Thread driver = new Thread(() -> drive(subscriber), "ooc-store-replay");
		driver.setDaemon(true);
		driver.start();
	}

	private void drive(Consumer<QueueCallback<T>> subscriber) {
		DMLRuntimeException failure = _failure;
		if(failure != null) {
			subscriber.accept(OOCStream.eos(failure));
			return;
		}
		try {
			while(_reader.hasNext()) {
				try(QueueCallback<T> callback = new MaterializedCallback<>(_reader.next())) {
					subscriber.accept(callback);
				}
			}
			_reader.close();
			subscriber.accept(OOCStream.eos(null));
		}
		catch(InterruptedException e) {
			Thread.currentThread().interrupt();
			subscriber.accept(OOCStream.eos(recordFailure(e)));
		}
		catch(RuntimeException e) {
			subscriber.accept(OOCStream.eos(recordFailure(e)));
		}
	}

	private synchronized DMLRuntimeException recordFailure(Exception error) {
		if(_failure == null)
			_failure = DMLRuntimeException.of(error);
		_reader.close();
		return _failure;
	}

	@Override
	public void propagateFailure(DMLRuntimeException failure) {
		recordFailure(failure);
	}

	@Override
	public OOCStream<T> getReadStream() {
		return this;
	}

	@Override
	public OOCStream<T> getWriteStream() {
		throw new UnsupportedOperationException("A store-backed stream has no write stream");
	}

	@Override
	public boolean hasStreamCache() {
		return false;
	}

	@Override
	public CachingStream getStreamCache() {
		return null;
	}

	@Override
	public boolean isProcessed() {
		return false;
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
}
