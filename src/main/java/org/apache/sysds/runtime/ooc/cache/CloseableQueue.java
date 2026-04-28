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

package org.apache.sysds.runtime.ooc.cache;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class CloseableQueue<T> {
	private final BlockingQueue<Object> queue = new LinkedBlockingQueue<>();
	private final Object POISON = new Object();  // sentinel
	private volatile boolean closed = false;

	public CloseableQueue() { }

	/**
	 * Enqueue if the queue is not closed.
	 * @return false if already closed
	 */
	public boolean enqueueIfOpen(T task) throws InterruptedException {
		if (task == null)
			throw new IllegalArgumentException("null tasks not allowed");
		synchronized (this) {
			if (closed)
				return false;
			queue.put(task);
		}
		return true;
	}

	@SuppressWarnings("unchecked")
	public T take() throws InterruptedException {
		if (closed && queue.isEmpty())
			return null;

		Object x = queue.take();

		if (x == POISON)
			return null;

		return (T) x;
	}

	/**
	 * Poll with max timeout.
	 * @return item, or null if:
	 *   - timeout, or
	 *   - queue has been closed and this consumer reached its poison pill
	 */
	@SuppressWarnings("unchecked")
	public T poll(long timeout, TimeUnit unit) throws InterruptedException {
		if (closed && queue.isEmpty())
			return null;

		Object x = queue.poll(timeout, unit);
		if (x == null)
			return null;          // timeout

		if (x == POISON)
			return null;

		return (T) x;
	}

	/**
	 * Close queue for N consumers.
	 * Each consumer will receive exactly one poison pill and then should stop.
	 */
	public boolean close() throws InterruptedException {
		synchronized (this) {
			if (closed)
				return false;           // idempotent
			closed = true;
		}
		queue.put(POISON);
		return true;
	}

	public synchronized boolean isFinished() {
		return closed && queue.isEmpty();
	}
}
