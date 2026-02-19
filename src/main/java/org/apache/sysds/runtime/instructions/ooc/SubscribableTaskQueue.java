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
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCGetStreamTypeMessage;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.LinkedList;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public class SubscribableTaskQueue<T> extends LocalTaskQueue<T> implements OOCStream<T> {

	private final AtomicInteger _availableCtr = new AtomicInteger(1);
	private final AtomicBoolean _closed = new AtomicBoolean(false);
	private final AtomicInteger _blockCount = new AtomicInteger(0);
	private CacheableData<?> _cdata;
	private volatile Consumer<QueueCallback<T>> _subscriber = null;
	private volatile CopyOnWriteArrayList<Consumer<OOCStreamMessage>> _upstreamMsgRelays = null;
	private volatile CopyOnWriteArrayList<Consumer<OOCStreamMessage>> _downstreamMsgRelays = null;
	private volatile BiFunction<Boolean, IndexRange, IndexRange> _ixTransform = null;
	private String _watchdogId;

	public SubscribableTaskQueue() {
		if (OOCWatchdog.WATCH) {
			_watchdogId = "STQ-" + hashCode();
			// Capture a short context to help identify origin
			OOCWatchdog.registerOpen(_watchdogId, "SubscribableTaskQueue@" + hashCode(), getCtxMsg(), this);
		}
	}

	private String getCtxMsg() {
		StackTraceElement[] st = new Exception().getStackTrace();
		// Skip the first few frames (constructor, createWritableStream, etc.)
		StringBuilder sb = new StringBuilder();
		int limit = Math.min(st.length, 7);
		for(int i = 2; i < limit; i++) {
			sb.append(st[i].getClassName()).append(".").append(st[i].getMethodName()).append(":")
				.append(st[i].getLineNumber());
			if(i < limit - 1)
				sb.append(" <- ");
		}
		return sb.toString();
	}

	@Override
	public void enqueue(T t) {
		if (t == NO_MORE_TASKS)
			throw new DMLRuntimeException("Cannot enqueue NO_MORE_TASKS item");

		int cnt = _availableCtr.incrementAndGet();

		if (cnt <= 1) { // Then the queue was already closed and we disallow further enqueues
			_availableCtr.decrementAndGet(); // Undo increment
			throw new DMLRuntimeException("Cannot enqueue into closed SubscribableTaskQueue");
		}

		_blockCount.incrementAndGet();

		Consumer<QueueCallback<T>> s = _subscriber;
		final Consumer<QueueCallback<T>> fS = s;

		if (fS != null) {
			fS.accept(new SimpleQueueCallback<>(t, _failure));
			onDeliveryFinished();
			return;
		}

		synchronized (this) {
			// Re-check that subscriber is really null to avoid race conditions
			if (_subscriber == null) {
				try {
					super.enqueueTask(t);
				}
				catch(InterruptedException e) {
					throw new DMLRuntimeException(e);
				}
				return;
			}
			// Otherwise do not insert and re-schedule subscriber invocation
			s = _subscriber;
		}

		// Last case if due to race a subscriber has been set
		s.accept(new SimpleQueueCallback<>(t, _failure));
		onDeliveryFinished();
	}

	@Override
	public synchronized void enqueueTask(T t) {
		enqueue(t);
	}

	@Override
	public T dequeue() {
		try {
			if (OOCWatchdog.WATCH)
				OOCWatchdog.addEvent(_watchdogId, "dequeue -- " + getCtxMsg());
			T deq = super.dequeueTask();
			if (deq != NO_MORE_TASKS)
				onDeliveryFinished();
			return deq;
		}
		catch(InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
	}

	@Override
	public synchronized T dequeueTask() {
		return dequeue();
	}

	@Override
	public synchronized void closeInput() {
		if (_closed.compareAndSet(false, true)) {
			super.closeInput();
			onDeliveryFinished();
			_upstreamMsgRelays = null;
			_downstreamMsgRelays = null;
		} else {
			throw new IllegalStateException("Multiple close input calls");
		}
	}

	private void validateBlockCountOnClose() {
		DataCharacteristics dc = getDataCharacteristics();
		if (dc != null && dc.dimsKnown() && dc.getBlocksize() > 0) {
			long expected = dc.getNumBlocks();
			if (expected >= 0 && _blockCount.get() != expected) {
				throw new DMLRuntimeException("OOCStream block count mismatch: expected "
					+ expected + " but saw " + _blockCount.get() + " (" + dc.getRows() + "x" + dc.getCols() + ")");
			}
		}
	}

	@Override
	public void setSubscriber(Consumer<QueueCallback<T>> subscriber) {
		if(subscriber == null)
			throw new IllegalArgumentException("Cannot set subscriber to null");

		LinkedList<T> data;

		synchronized(this) {
			if(_subscriber != null)
				throw new DMLRuntimeException("Cannot set multiple subscribers");
			_subscriber = subscriber;
			if(_failure != null)
				throw _failure;
			data = _data;
			_data = new LinkedList<>();
		}

		for (T t : data) {
			subscriber.accept(new SimpleQueueCallback<>(t, _failure));
			onDeliveryFinished();
		}
	}

	@SuppressWarnings("unchecked")
	private void onDeliveryFinished() {
		int ctr = _availableCtr.decrementAndGet();

		if (ctr == 0) {
			validateBlockCountOnClose();
			Consumer<QueueCallback<T>> s = _subscriber;
			if (s != null)
				s.accept(new SimpleQueueCallback<>((T) LocalTaskQueue.NO_MORE_TASKS, _failure));

			if (OOCWatchdog.WATCH)
				OOCWatchdog.registerClose(_watchdogId);
		}
	}

	@Override
	public synchronized void propagateFailure(DMLRuntimeException re) {
		super.propagateFailure(re);
		Consumer<QueueCallback<T>> s = _subscriber;
		if(s != null)
			s.accept(new SimpleQueueCallback<>(null, re));
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
	public void messageUpstream(OOCStreamMessage msg) {
		if(msg.isCancelled())
			return;
		msg.addIXTransform(_ixTransform);
		if (msg.isCancelled())
			return;
		if (msg instanceof OOCGetStreamTypeMessage) {
			if (_cdata != null)
				((OOCGetStreamTypeMessage) msg).setInMemoryType();
			return;
		}
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _upstreamMsgRelays;
		if(relays != null) {
			for (Consumer<OOCStreamMessage> relay : relays) {
				if (msg.isCancelled())
					break;
				relay.accept(msg);
			}
		}
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		if(!msg.isCancelled())
			return;
		msg.addIXTransform(_ixTransform);
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamMsgRelays;
		if(relays != null) {
			for (Consumer<OOCStreamMessage> relay : relays) {
				if (msg.isCancelled())
					break;
				relay.accept(msg);
			}
		}
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
	public DataCharacteristics getDataCharacteristics() {
		return _cdata == null ? null : _cdata.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _cdata;
	}

	@Override
	public void setData(CacheableData<?> data) {
		if(_cdata == null && _closed.get())
			System.out.println("[WARN] Data type was defined after closing, which may bypass validation checks");
		_cdata = data;
	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		addUpstreamMessageRelay(relay);
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		addDownstreamMessageRelay(relay);
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		if(relay == null)
			throw new IllegalArgumentException("Cannot set upstream relay to null");
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _upstreamMsgRelays;
		if(relays == null) {
			synchronized(this) {
				if(_upstreamMsgRelays == null)
					_upstreamMsgRelays = new CopyOnWriteArrayList<>();
				relays = _upstreamMsgRelays;
			}
		}
		relays.add(0, relay);
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		if(relay == null)
			throw new IllegalArgumentException("Cannot set downstream relay to null");
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamMsgRelays;
		if(relays == null) {
			synchronized(this) {
				if(_downstreamMsgRelays == null)
					_downstreamMsgRelays = new CopyOnWriteArrayList<>();
				relays = _downstreamMsgRelays;
			}
		}
		relays.add(0, relay);
	}

	@Override
	public void clearUpstreamMessageRelays() {
		_upstreamMsgRelays = null;
	}

	@Override
	public void clearDownstreamMessageRelays() {
		_downstreamMsgRelays = null;
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		_ixTransform = transform;
	}

	@Override
	public String toString() {
		return "STQ-" + hashCode();
	}
}
