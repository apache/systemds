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

public class SubscribableTaskQueue<T> extends LocalTaskQueue<T> implements OOCStream<T> {
	private Runnable _subscriber;

	@Override
	public void enqueue(T t) {
		try {
			super.enqueueTask(t);
		}
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		if(_subscriber != null)
			_subscriber.run();
	}

	@Override
	public T dequeue() {
		try {
			return super.dequeueTask();
		}
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
	}

	@Override
	public synchronized void closeInput() {
		super.closeInput();

		if(_subscriber != null) {
			_subscriber.run();
			_subscriber = null;
		}
	}

	@Override
	public LocalTaskQueue<T> toLocalTaskQueue() {
		return this;
	}

	@Override
	public OOCStream<T> getReadStream() {
		return this;
	}

	@Override
	public OOCStream<T> getWriteStream() {
		return this;
	}

	@Override
	public void setSubscriber(Runnable subscriber) {
		int queueSize;

		synchronized (this) {
			if(_subscriber != null)
				throw new DMLRuntimeException("Cannot set multiple subscribers");

			_subscriber = subscriber;
			queueSize = _data.size();
			queueSize += _closedInput ? 1 : 0; // To trigger the NO_MORE_TASK element
		}

		for (int i = 0; i < queueSize; i++)
			subscriber.run();
	}

	@Override
	public synchronized void propagateFailure(DMLRuntimeException re) {
		super.propagateFailure(re);

		if(_subscriber != null)
			_subscriber.run();
	}

	@Override
	public boolean hasStreamCache() {
		return false;
	}

	@Override
	public CachingStream getStreamCache() {
		return null;
	}
}
