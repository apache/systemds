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

import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Small future implementation for OOC hot paths. It supports multiple synchronous subscribers without
 * the completion-stage support of {@link java.util.concurrent.CompletableFuture}.
 */
public class OOCFuture<T> {
	private Subscriber<T> _subscribers;
	private T _value;
	private Throwable _error;
	private boolean _done;

	public static <T> OOCFuture<T> completed(T value) {
		OOCFuture<T> future = new OOCFuture<>();
		future._value = value;
		future._done = true;
		return future;
	}

	public static <T> OOCFuture<T> failed(Throwable error) {
		OOCFuture<T> future = new OOCFuture<>();
		future._error = error;
		future._done = true;
		return future;
	}

	public boolean complete(T value) {
		return finish(value, null);
	}

	public boolean completeExceptionally(Throwable error) {
		if(error == null)
			throw new NullPointerException("error");
		return finish(null, error);
	}

	public void thenAccept(Consumer<? super T> action) {
		subscribe(null, action, null);
	}

	public void whenComplete(BiConsumer<? super T, ? super Throwable> action) {
		subscribe(null, null, action);
	}

	public <R> OOCFuture<R> map(Function<? super T, ? extends R> mapper) {
		return new MappedFuture<>(this, mapper);
	}

	public synchronized boolean isDone() {
		return _done;
	}

	public synchronized T getNow(T fallback) {
		if(!_done)
			return fallback;
		if(_error != null)
			throw new CompletionException(_error);
		return _value;
	}

	public T get() throws InterruptedException, ExecutionException {
		synchronized(this) {
			while(!_done)
				wait();
			if(_error != null)
				throw new ExecutionException(_error);
			return _value;
		}
	}

	public T get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
		long remaining = unit.toNanos(timeout);
		long deadline = System.nanoTime() + remaining;
		synchronized(this) {
			while(!_done) {
				if(remaining <= 0)
					throw new TimeoutException();
				TimeUnit.NANOSECONDS.timedWait(this, remaining);
				remaining = deadline - System.nanoTime();
			}
			if(_error != null)
				throw new ExecutionException(_error);
			return _value;
		}
	}

	private <R> void subscribe(Function<? super T, ? extends R> mapper, Consumer<? super R> action,
		BiConsumer<? super R, ? super Throwable> completion) {
		T value;
		Throwable error;
		synchronized(this) {
			if(!_done) {
				_subscribers = new Subscriber<>(mapper, action, completion, _subscribers);
				return;
			}
			value = _value;
			error = _error;
		}
		accept(mapper, action, completion, value, error);
	}

	private boolean finish(T value, Throwable error) {
		Subscriber<T> subscribers;
		synchronized(this) {
			if(_done)
				return false;
			_value = value;
			_error = error;
			_done = true;
			subscribers = _subscribers;
			_subscribers = null;
			notifyAll();
		}
		while(subscribers != null) {
			Subscriber<T> next = subscribers.next;
			subscribers.accept(value, error);
			subscribers = next;
		}
		return true;
	}

	private static <T, R> void accept(Function<? super T, ? extends R> mapper, Consumer<? super R> action,
		BiConsumer<? super R, ? super Throwable> completion, T value, Throwable error) {
		R result = null;
		Throwable resultError = error;
		if(resultError == null) {
			try {
				@SuppressWarnings("unchecked")
				R mapped = mapper == null ? (R)value : mapper.apply(value);
				result = mapped;
			}
			catch(Throwable t) {
				resultError = t;
			}
		}
		try {
			if(completion != null)
				completion.accept(result, resultError);
			else if(resultError == null)
				action.accept(result);
		}
		catch(Throwable ignored) {
			// Subscribers are independent; one failed callback must not prevent the remaining notifications.
		}
	}

	private static final class Subscriber<T> {
		private final Function<? super T, ?> mapper;
		private final Consumer<Object> action;
		private final BiConsumer<Object, Throwable> completion;
		private final Subscriber<T> next;

		@SuppressWarnings("unchecked")
		private <R> Subscriber(Function<? super T, ? extends R> mapper, Consumer<? super R> action,
			BiConsumer<? super R, ? super Throwable> completion, Subscriber<T> next) {
			this.mapper = mapper;
			this.action = (Consumer<Object>)action;
			this.completion = (BiConsumer<Object, Throwable>)(BiConsumer<?, ?>)completion;
			this.next = next;
		}

		private void accept(T value, Throwable error) {
			OOCFuture.accept(mapper, action, completion, value, error);
		}
	}

	private static final class MappedFuture<S, T> extends OOCFuture<T> {
		private final OOCFuture<S> source;
		private final Function<? super S, ? extends T> mapper;

		private MappedFuture(OOCFuture<S> source, Function<? super S, ? extends T> mapper) {
			this.source = source;
			this.mapper = mapper;
		}

		@Override
		public boolean complete(T value) {
			throw new UnsupportedOperationException("Cannot complete a mapped OOCFuture");
		}

		@Override
		public boolean completeExceptionally(Throwable error) {
			throw new UnsupportedOperationException("Cannot complete a mapped OOCFuture");
		}

		@Override
		public void thenAccept(Consumer<? super T> action) {
			source.subscribe(mapper, action, null);
		}

		@Override
		public void whenComplete(BiConsumer<? super T, ? super Throwable> action) {
			source.subscribe(mapper, null, action);
		}

		@Override
		public <R> OOCFuture<R> map(Function<? super T, ? extends R> nextMapper) {
			return new MappedFuture<>(source, value -> nextMapper.apply(mapper.apply(value)));
		}

		@Override
		public boolean isDone() {
			return source.isDone();
		}

		@Override
		public T getNow(T fallback) {
			if(!source.isDone())
				return fallback;
			try {
				return mapper.apply(source.getNow(null));
			}
			catch(CompletionException ex) {
				throw ex;
			}
			catch(Throwable t) {
				throw new CompletionException(t);
			}
		}

		@Override
		public T get() throws InterruptedException, ExecutionException {
			try {
				return mapper.apply(source.get());
			}
			catch(InterruptedException | ExecutionException ex) {
				throw ex;
			}
			catch(Throwable t) {
				throw new ExecutionException(t);
			}
		}

		@Override
		public T get(long timeout, TimeUnit unit)
			throws InterruptedException, ExecutionException, TimeoutException {
			try {
				return mapper.apply(source.get(timeout, unit));
			}
			catch(InterruptedException | ExecutionException | TimeoutException ex) {
				throw ex;
			}
			catch(Throwable t) {
				throw new ExecutionException(t);
			}
		}
	}
}
