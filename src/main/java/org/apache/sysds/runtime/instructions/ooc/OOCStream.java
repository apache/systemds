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

import java.util.function.Consumer;

public interface OOCStream<T> extends OOCStreamable<T> {
	static <T> QueueCallback<T> eos(DMLRuntimeException e) {
		return new SimpleQueueCallback<>(null, e);
	}

	void enqueue(T t);

	T dequeue();

	void closeInput();

	void propagateFailure(DMLRuntimeException re);

	boolean hasStreamCache();

	CachingStream getStreamCache();

	/**
	 * Registers a new subscriber that consumes the stream.
	 * While there is no guarantee for any specific order, the closing item LocalTaskQueue.NO_MORE_TASKS
	 * is guaranteed to be invoked after every other item has finished processing. Thus, the NO_MORE_TASKS
	 * callback can be used to free dependent resources and close output streams.
	 */
	void setSubscriber(Consumer<QueueCallback<T>> subscriber);

	interface QueueCallback<T> extends AutoCloseable {
		T get();

		/**
		 * Keeps the callback item pinned in memory until the returned callback is also closed.
		 */
		QueueCallback<T> keepOpen();

		void close();

		void fail(DMLRuntimeException failure);

		boolean isEos();
	}

	class SimpleQueueCallback<T> implements QueueCallback<T> {
		private final T _result;
		private DMLRuntimeException _failure;

		public SimpleQueueCallback(T result, DMLRuntimeException failure) {
			this._result = result;
			this._failure = failure;
		}

		@Override
		public T get() {
			if (_failure != null)
				throw _failure;
			return _result;
		}

		@Override
		public QueueCallback<T> keepOpen() {
			return this;
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			this._failure = failure;
		}

		@Override
		public void close() {}

		@Override
		public boolean isEos() {
			return get() == null;
		}
	}
}
