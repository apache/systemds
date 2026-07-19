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

package org.apache.sysds.runtime.ooc.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.stream.TaskContext;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.Statistics;

public final class OOCInstructionUtils {
	public static final boolean ALLOW_PIPELINING = true;
	public static final ExecutorService COMPUTE_EXECUTOR = CommonThreadPool.get();
	private static final AtomicInteger COMPUTE_IN_FLIGHT = new AtomicInteger();
	private static final int COMPUTE_BACKPRESSURE_THRESHOLD = 100;

	public static int getComputeInFlight() {
		return COMPUTE_IN_FLIGHT.get();
	}

	public static int getComputeBackpressureThreshold() {
		return COMPUTE_BACKPRESSURE_THRESHOLD;
	}

	public static <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue,
		Consumer<OOCStream.QueueCallback<T>> consumer, StreamContext context) {
		return submitOOCTasks(List.of(queue), (i, callback) -> consumer.accept(callback), null, null, context);
	}

	public static <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue,
		Consumer<OOCStream.QueueCallback<T>> consumer, Function<OOCStream.QueueCallback<T>, Boolean> predicate,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed, StreamContext context) {
		return submitOOCTasks(List.of(queue), (i, callback) -> consumer.accept(callback),
			(i, callback) -> predicate.apply(callback), onNotProcessed, context);
	}

	public static <T> CompletableFuture<Void> submitOOCTasks(List<OOCStream<T>> queues,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer, StreamContext context) {
		return submitOOCTasks(queues, consumer, null, null, context);
	}

	public static <T> CompletableFuture<Void> submitOOCTasks(List<OOCStream<T>> queues,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer,
		BiFunction<Integer, OOCStream.QueueCallback<T>, Boolean> predicate,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed, StreamContext context) {
		context.addInStream(queues.toArray(OOCStream[]::new));
		if(!context.outStreamsDefined())
			throw new IllegalArgumentException("Explicit specification of all output streams is required before "
				+ "submitting tasks. If no output streams are present use addOutStream().");

		List<AtomicInteger> activeTaskCounters = new ArrayList<>(queues.size());
		List<OOCFuture<Void>> futures = new ArrayList<>(queues.size());
		for(int i = 0; i < queues.size(); i++) {
			activeTaskCounters.add(new AtomicInteger(1));
			futures.add(new OOCFuture<>());
		}

		CompletableFuture<Void> globalFuture = new CompletableFuture<>();
		AtomicInteger remaining = new AtomicInteger(futures.size());
		if(futures.isEmpty())
			globalFuture.complete(null);
		for(OOCFuture<Void> future : futures)
			future.whenComplete((result, error) -> {
				if(error != null)
					globalFuture.completeExceptionally(error);
				else if(remaining.decrementAndGet() == 0)
					globalFuture.complete(null);
			});
		StreamContext streamContext = context.copy();
		for(int i = 0; i < queues.size(); i++)
			subscribe(queues.get(i), i, consumer, predicate, onNotProcessed, activeTaskCounters.get(i), futures.get(i),
				globalFuture, streamContext);

		return globalFuture.handle((result, error) -> {
			if(error != null) {
				for(OOCFuture<Void> future : futures)
					if(!future.isDone())
						future.completeExceptionally(error);
			}
			streamContext.clear();
			return null;
		});
	}

	private static <T> void subscribe(OOCStream<T> queue, int streamIndex,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer,
		BiFunction<Integer, OOCStream.QueueCallback<T>, Boolean> predicate,
		BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed, AtomicInteger activeTaskCounter,
		OOCFuture<Void> future, CompletableFuture<Void> globalFuture, StreamContext context) {
		AtomicBoolean closed = new AtomicBoolean();
		queue.setSubscriber(guard(callback -> {
			long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
			try(callback) {
				if(callback.isEos()) {
					if(!closed.compareAndSet(false, true))
						throw new DMLRuntimeException(
							"Race condition observed: NO_MORE_TASKS callback has been triggered more than once");
					if(activeTaskCounter.decrementAndGet() == 0)
						future.complete(null);
					return;
				}

				Consumer<OOCStream.QueueCallback<T>> process = item -> {
					if(predicate != null && !predicate.apply(streamIndex, item)) {
						if(onNotProcessed != null)
							onNotProcessed.accept(streamIndex, item);
						return;
					}
					if(future.isDone()) {
						if(onNotProcessed != null)
							onNotProcessed.accept(streamIndex, item);
						return;
					}

					activeTaskCounter.incrementAndGet();
					OOCStream.QueueCallback<T> pinned = item.keepOpen();
					submit(() -> {
						long taskStartTime = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
						try(pinned) {
							consumer.accept(streamIndex, pinned);
							if(activeTaskCounter.decrementAndGet() == 0)
								TaskContext.defer(() -> future.complete(null));
						}
						finally {
							recordStatistics(context, globalFuture, taskStartTime);
							recordEvent(context, taskStartTime);
						}
					}, future, context);
				};

				if(callback instanceof OOCStream.GroupQueueCallback<?>) {
					OOCStream.GroupQueueCallback<T> group = (OOCStream.GroupQueueCallback<T>) callback;
					if(future.isDone()) {
						for(int index = 0; index < group.size(); index++) {
							try(OOCStream.QueueCallback<T> item = group.getCallback(index)) {
								if(onNotProcessed != null)
									onNotProcessed.accept(streamIndex, item);
							}
						}
						return;
					}

					activeTaskCounter.incrementAndGet();
					OOCStream.GroupQueueCallback<T> pinned = (OOCStream.GroupQueueCallback<T>) group.keepOpen();
					submit(() -> {
						long taskStartTime = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
						try(pinned) {
							for(int index = 0; index < pinned.size(); index++) {
								try(OOCStream.QueueCallback<T> item = pinned.getCallback(index)) {
									process.accept(item);
								}
							}
							if(activeTaskCounter.decrementAndGet() == 0)
								TaskContext.defer(() -> future.complete(null));
						}
						finally {
							recordStatistics(context, globalFuture, taskStartTime);
							recordEvent(context, taskStartTime);
						}
					}, future, context);
				}
				else
					process.accept(callback);

				if(closed.get())
					throw new DMLRuntimeException("Race condition observed");
			}
			catch(RuntimeException error) {
				context.failAll(DMLRuntimeException.of(error));
				throw error;
			}
			finally {
				recordStatistics(context, globalFuture, startTime);
			}
		}, context));
	}

	public static OOCFuture<Void> submitOOCTask(Runnable task, StreamContext context) {
		// May be blocking tasks, thus should not run on default executor pool
		ExecutorService pool = CommonThreadPool.getDynamicPool();
		OOCFuture<Void> future = new OOCFuture<>();
		COMPUTE_IN_FLIGHT.incrementAndGet();
		try {
			pool.submit(task(() -> {
				long startTime = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
				try {
					task.run();
					future.complete(null);
					context.clear();
					if(DMLScript.STATISTICS && context.getExtendedOpcode() != null)
						Statistics.maintainOOCHeavyHitter(context.getExtendedOpcode(), System.nanoTime() - startTime);
					recordEvent(context, startTime);
				}
				finally {
					COMPUTE_IN_FLIGHT.decrementAndGet();
				}
			}, future, context));
		}
		catch(RuntimeException error) {
			COMPUTE_IN_FLIGHT.decrementAndGet();
			throw DMLRuntimeException.of(error);
		}
		return future;
	}

	private static void submit(Runnable runnable, OOCFuture<Void> future, StreamContext context) {
		COMPUTE_IN_FLIGHT.incrementAndGet();
		try {
			COMPUTE_EXECUTOR.submit(task(() -> {
				try {
					runnable.run();
				}
				finally {
					COMPUTE_IN_FLIGHT.decrementAndGet();
				}
			}, future, context));
		}
		catch(RuntimeException error) {
			COMPUTE_IN_FLIGHT.decrementAndGet();
			throw error;
		}
	}

	private static Runnable task(Runnable runnable, OOCFuture<Void> future, StreamContext context) {
		return () -> {
			boolean setContext = TaskContext.getContext() == null;
			if(setContext)
				TaskContext.setContext(new TaskContext());
			long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
			try {
				runnable.run();
				if(setContext) {
					while(TaskContext.runDeferred()) {
					}
				}
			}
			catch(RuntimeException error) {
				DMLRuntimeException failure = DMLRuntimeException.of(error);
				context.failAll(failure);
				if(future != null)
					future.completeExceptionally(failure);
				throw failure;
			}
			finally {
				if(setContext)
					TaskContext.clearContext();
				if(DMLScript.STATISTICS)
					context.getLocalStatisticsLongAdder().add(System.nanoTime() - startTime);
			}
		};
	}

	private static <T> Consumer<OOCStream.QueueCallback<T>> guard(Consumer<OOCStream.QueueCallback<T>> consumer,
		StreamContext context) {
		return callback -> {
			try {
				consumer.accept(callback);
			}
			catch(RuntimeException error) {
				DMLRuntimeException failure = DMLRuntimeException.of(error);
				context.failAll(failure);
				throw failure;
			}
		};
	}

	private static void recordStatistics(StreamContext context, CompletableFuture<Void> globalFuture, long startTime) {
		if(!DMLScript.STATISTICS)
			return;
		context.getLocalStatisticsLongAdder().add(System.nanoTime() - startTime);
		if(globalFuture.isDone() && context.getExtendedOpcode() != null) {
			Statistics.maintainOOCHeavyHitter(context.getExtendedOpcode(), context.getLocalStatisticsLongAdder().sum());
			context.getLocalStatisticsLongAdder().reset();
		}
	}

	private static void recordEvent(StreamContext context, long startTime) {
		if(DMLScript.OOC_LOG_EVENTS && context.getCallerId() != 0)
			OOCEventLog.onComputeEvent(context.getCallerId(), startTime, System.nanoTime());
	}
}
