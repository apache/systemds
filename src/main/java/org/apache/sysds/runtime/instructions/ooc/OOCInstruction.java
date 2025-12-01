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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.OOCJoin;
import org.apache.sysds.utils.Statistics;
import scala.Tuple4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;

public abstract class OOCInstruction extends Instruction {
	protected static final Log LOG = LogFactory.getLog(OOCInstruction.class.getName());
	private static final AtomicInteger nextStreamId = new AtomicInteger(0);
	private long nanoTime;

	public enum OOCType {
		Reblock, Tee, Binary, Ternary, Unary, AggregateUnary, AggregateBinary, AggregateTernary, MAPMM, MMTSJ,
		Reorg, CM, Ctable, MatrixIndexing, ParameterizedBuiltin, Rand
	}

	protected final OOCInstruction.OOCType _ooctype;
	protected final boolean _requiresLabelUpdate;
	protected Set<OOCStream<?>> _inQueues;
	protected Set<OOCStream<?>> _outQueues;
	private boolean _failed;
	private LongAdder _localStatisticsAdder;
	public final int _callerId;

	protected OOCInstruction(OOCInstruction.OOCType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected OOCInstruction(OOCInstruction.OOCType type, Operator op, String opcode, String istr) {
		super(op);
		_ooctype = type;
		instString = istr;
		instOpcode = opcode;

		_requiresLabelUpdate = super.requiresLabelUpdate();
		_failed = false;

		if (DMLScript.STATISTICS)
			_localStatisticsAdder = new LongAdder();
		_callerId = DMLScript.OOC_LOG_EVENTS ? OOCEventLog.registerCaller(getExtendedOpcode() + "_" + hashCode()) : 0;
	}

	@Override
	public IType getType() {
		return IType.OUT_OF_CORE;
	}

	public OOCInstruction.OOCType getOOCInstructionType() {
		return _ooctype;
	}

	@Override
	public boolean requiresLabelUpdate() {
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		if (DMLScript.OOC_LOG_EVENTS)
			nanoTime = System.nanoTime();
		// TODO
		return super.preprocessInstruction(ec);
	}

	@Override
	public abstract void processInstruction(ExecutionContext ec);

	@Override
	public void postprocessInstruction(ExecutionContext ec) {
		if(DMLScript.LINEAGE_DEBUGGER)
			ec.maintainLineageDebuggerInfo(this);
		if (DMLScript.OOC_LOG_EVENTS)
			OOCEventLog.onComputeEvent(_callerId, nanoTime, System.nanoTime());
	}

	protected void addInStream(OOCStream<?>... queue) {
		if (_inQueues == null)
			_inQueues = new HashSet<>();
		_inQueues.addAll(List.of(queue));
	}

	protected void addOutStream(OOCStream<?>... queue) {
		// Currently same behavior as addInQueue
		if (_outQueues == null)
			_outQueues = new HashSet<>();
		_outQueues.addAll(List.of(queue));
	}

	protected <T> OOCStream<T> createWritableStream() {
		return new SubscribableTaskQueue<>();
	}

	protected <T> CompletableFuture<Void> filterOOC(OOCStream<T> qIn, Consumer<T> processor, Function<T, Boolean> predicate, Runnable finalizer) {
		return filterOOC(qIn, processor, predicate, finalizer, null);
	}

	protected <T> CompletableFuture<Void> filterOOC(OOCStream<T> qIn, Consumer<T> processor, Function<T, Boolean> predicate, Runnable finalizer, Consumer<T> onNotProcessed) {
		if (_inQueues == null || _outQueues == null)
			throw new NotImplementedException("filterOOC requires manual specification of all input and output streams for error propagation");

		return submitOOCTasks(qIn, c -> processor.accept(c.get()), finalizer, p -> predicate.apply(p.get()), onNotProcessed != null ? (i, tmp) -> onNotProcessed.accept(tmp.get()) : null);
	}

	protected <T, R> CompletableFuture<Void> mapOOC(OOCStream<T> qIn, OOCStream<R> qOut, Function<T, R> mapper) {
		addInStream(qIn);
		addOutStream(qOut);

		return submitOOCTasks(qIn, tmp -> {
			try (tmp) {
				R r = mapper.apply(tmp.get());
				qOut.enqueue(r);
			} catch (Exception e) {
				throw e instanceof DMLRuntimeException ? (DMLRuntimeException) e : new DMLRuntimeException(e);
			}
		}, qOut::closeInput);
	}

	protected <R, P> CompletableFuture<Void> broadcastJoinOOC(OOCStream<IndexedMatrixValue> qIn, OOCStream<IndexedMatrixValue> broadcast, OOCStream<R> qOut, BiFunction<IndexedMatrixValue, BroadcastedElement, R> mapper, Function<IndexedMatrixValue, P> on) {
		addInStream(qIn, broadcast);
		addOutStream(qOut);

		boolean explicitLeftCaching = !qIn.hasStreamCache();
		boolean explicitRightCaching = !broadcast.hasStreamCache();
		CachingStream leftCache = explicitLeftCaching ? new CachingStream(new SubscribableTaskQueue<>()) : qIn.getStreamCache();
		CachingStream rightCache = explicitRightCaching ? new CachingStream(new SubscribableTaskQueue<>()) : broadcast.getStreamCache();
		leftCache.activateIndexing();
		rightCache.activateIndexing();

		if (!explicitLeftCaching)
			leftCache.incrSubscriberCount(1); // Prevent early block deletion as we may read elements twice

		if (!explicitRightCaching)
			rightCache.incrSubscriberCount(1);

		Map<P, List<MatrixIndexes>> availableLeftInput = new ConcurrentHashMap<>();
		Map<P, BroadcastedElement> availableBroadcastInput = new ConcurrentHashMap<>();

		OOCStream<Tuple4<P, OOCStream.QueueCallback<IndexedMatrixValue>, OOCStream.QueueCallback<IndexedMatrixValue>, BroadcastedElement>> broadcastingQueue = createWritableStream();
		AtomicInteger waitCtr = new AtomicInteger(1);
		CompletableFuture<Void> fut1 = new CompletableFuture<>();

		submitOOCTasks(List.of(qIn, broadcast), (i, tmp) -> {
			try (tmp) {
				P key = on.apply(tmp.get());

				if(i == 0) { // qIn stream
					BroadcastedElement b = availableBroadcastInput.get(key);

					if(b == null) {
						// Matching broadcast element is not available -> cache element
						availableLeftInput.compute(key, (k, v) -> {
							if(v == null)
								v = new ArrayList<>();
							v.add(tmp.get().getIndexes());
							return v;
						});

						if(explicitLeftCaching)
							leftCache.getWriteStream().enqueue(tmp.get());
					}
					else {
						waitCtr.incrementAndGet();

						OOCCacheManager.requestManyBlocks(
							List.of(leftCache.peekCachedBlockKey(tmp.get().getIndexes()), rightCache.peekCachedBlockKey(b.idx)))
							.whenComplete((items, err) -> {
								try {
									broadcastingQueue.enqueue(new Tuple4<>(key, items.get(0).keepOpen(), items.get(1).keepOpen(), b));
								} finally {
									items.forEach(OOCStream.QueueCallback::close);
								}
							});
					}
				}
				else { // broadcast stream
					if(explicitRightCaching)
						rightCache.getWriteStream().enqueue(tmp.get());

					BroadcastedElement b = new BroadcastedElement(tmp.get().getIndexes());
					availableBroadcastInput.put(key, b);

					List<MatrixIndexes> queued = availableLeftInput.remove(key);

					if(queued != null) {
						for(MatrixIndexes idx : queued) {
							waitCtr.incrementAndGet();

							OOCCacheManager.requestManyBlocks(
								List.of(leftCache.peekCachedBlockKey(idx), rightCache.peekCachedBlockKey(tmp.get().getIndexes())))
								.whenComplete((items, err) -> {
									try{
										broadcastingQueue.enqueue(new Tuple4<>(key, items.get(0).keepOpen(), items.get(1).keepOpen(), b));
									} finally {
										items.forEach(OOCStream.QueueCallback::close);
									}
								});
						}
					}
				}
			}
		}, () -> {
			fut1.complete(null);
			if (waitCtr.decrementAndGet() == 0)
				broadcastingQueue.closeInput();
		});

		CompletableFuture<Void> fut2 = new CompletableFuture<>();
		submitOOCTasks(List.of(broadcastingQueue), (i, tpl) -> {
			try (tpl) {
				final BroadcastedElement b = tpl.get()._4();
				final OOCStream.QueueCallback<IndexedMatrixValue> lValue = tpl.get()._2();
				final OOCStream.QueueCallback<IndexedMatrixValue> bValue = tpl.get()._3();

				try (lValue; bValue) {
					b.value = bValue.get();
					qOut.enqueue(mapper.apply(lValue.get(), b));
					leftCache.incrProcessingCount(leftCache.findCachedIndex(lValue.get().getIndexes()), 1);

					if(b.canRelease()) {
						availableBroadcastInput.remove(tpl.get()._1());

						if(!explicitRightCaching)
							rightCache.incrProcessingCount(rightCache.findCachedIndex(b.idx),
								1); // Correct for incremented subscriber count to allow block deletion
					}
				}

				if(waitCtr.decrementAndGet() == 0)
					broadcastingQueue.closeInput();
			}
		}, () -> fut2.complete(null));

		if (explicitLeftCaching)
			leftCache.scheduleDeletion();
		if (explicitRightCaching)
			rightCache.scheduleDeletion();

		CompletableFuture<Void> fut = CompletableFuture.allOf(fut1, fut2);
		fut.whenComplete((res, t) -> {
			availableBroadcastInput.forEach((k, v) -> {
				rightCache.incrProcessingCount(rightCache.findCachedIndex(v.idx), 1);
			});
			availableBroadcastInput.clear();
			qOut.closeInput();
		});

		return fut;
	}

	protected static class BroadcastedElement {
		private final MatrixIndexes idx;
		private IndexedMatrixValue value;
		private boolean release;
		private int processCtr;

		public BroadcastedElement(MatrixIndexes idx) {
			this.idx = idx;
			this.release = false;
		}

		public synchronized void release() {
			release = true;
		}

		public synchronized boolean canRelease() {
			return release;
		}

		public synchronized int incrProcessCtrAndGet() {
			processCtr++;
			return processCtr;
		}

		public MatrixIndexes getIndex() {
			return idx;
		}

		public IndexedMatrixValue getValue() {
			return value;
		}
	}

	protected <T, R, P> CompletableFuture<Void> joinOOC(OOCStream<T> qIn1, OOCStream<T> qIn2, OOCStream<R> qOut, BiFunction<T, T, R> mapper, Function<T, P> on) {
		return joinOOC(qIn1, qIn2, qOut, mapper, on, on);
	}

	@SuppressWarnings("unchecked")
	protected <T, R, P> CompletableFuture<Void> joinOOC(List<OOCStream<T>> qIn, OOCStream<R> qOut, Function<List<T>, R> mapper, List<Function<T, P>> on) {
		if (qIn == null || on == null || qIn.size() != on.size())
			throw new DMLRuntimeException("joinOOC(list) requires the same number of streams and key functions.");

		addInStream(qIn.toArray(OOCStream[]::new));
		addOutStream(qOut);

		final int n = qIn.size();

		CachingStream[] caches = new CachingStream[n];
		boolean[] explicitCaching = new boolean[n];

		for (int i = 0; i < n; i++) {
			OOCStream<T> s = qIn.get(i);
			explicitCaching[i] = !s.hasStreamCache();
			caches[i] = explicitCaching[i] ? new CachingStream((OOCStream<IndexedMatrixValue>) s) : s.getStreamCache();
			caches[i].activateIndexing();
			// One additional consumption for the materialization when emitting
			caches[i].incrSubscriberCount(1);
		}

		Map<P, MatrixIndexes[]> seen = new ConcurrentHashMap<>();

		CompletableFuture<Void> future = submitOOCTasks(
			Arrays.stream(caches).map(CachingStream::getReadStream).collect(java.util.stream.Collectors.toList()),
			(i, tmp) -> {
				Function<T, P> keyFn = on.get(i);
				P key = keyFn.apply((T)tmp.get());
				MatrixIndexes idx = tmp.get().getIndexes();

				MatrixIndexes[] arr = seen.computeIfAbsent(key, k -> new MatrixIndexes[n]);
				boolean ready;
				synchronized (arr) {
					arr[i] = idx;
					ready = true;
					for (MatrixIndexes ix : arr) {
						if (ix == null) {
							ready = false;
							break;
						}
					}
				}

				if (!ready || !seen.remove(key, arr))
					return;

				List<OOCStream.QueueCallback<T>> values = new java.util.ArrayList<>(n);
				try {
					for(int j = 0; j < n; j++)
						values.add((OOCStream.QueueCallback<T>) caches[j].findCached(arr[j]));

					qOut.enqueue(mapper.apply(values.stream().map(OOCStream.QueueCallback::get).toList()));
				} finally {
					values.forEach(OOCStream.QueueCallback::close);
				}
			}, qOut::closeInput);

		for (int i = 0; i < n; i++) {
			if (explicitCaching[i])
				caches[i].scheduleDeletion();
		}

		return future;
	}

	@SuppressWarnings("unchecked")
	protected <T, R, P> CompletableFuture<Void> joinOOC(OOCStream<T> qIn1, OOCStream<T> qIn2, OOCStream<R> qOut, BiFunction<T, T, R> mapper, Function<T, P> onLeft, Function<T, P> onRight) {
		addInStream(qIn1, qIn2);
		addOutStream(qOut);

		final CompletableFuture<Void> future = new CompletableFuture<>();

		boolean explicitLeftCaching = !qIn1.hasStreamCache();
		boolean explicitRightCaching = !qIn2.hasStreamCache();

		// We need to construct our own stream to properly manage the cached items in the hash join
		CachingStream leftCache = explicitLeftCaching ? new CachingStream((OOCStream<IndexedMatrixValue>) qIn1) : qIn1.getStreamCache();
		CachingStream rightCache = explicitRightCaching ? new CachingStream((OOCStream<IndexedMatrixValue>) qIn2) : qIn2.getStreamCache();
		leftCache.activateIndexing();
		rightCache.activateIndexing();

		leftCache.incrSubscriberCount(1);
		rightCache.incrSubscriberCount(1);

		final OOCJoin<P, MatrixIndexes> join = new OOCJoin<>((idx, left, right) -> {
			OOCStream.QueueCallback<T> leftObj = (OOCStream.QueueCallback<T>) leftCache.findCached(left);
			OOCStream.QueueCallback<T> rightObj = (OOCStream.QueueCallback<T>) rightCache.findCached(right);
			try (leftObj; rightObj) {
				qOut.enqueue(mapper.apply(leftObj.get(), rightObj.get()));
			}
		});

		submitOOCTasks(List.of(leftCache.getReadStream(), rightCache.getReadStream()), (i, tmp) -> {
			try (tmp) {
				if(i == 0)
					join.addLeft(onLeft.apply((T) tmp.get()), tmp.get().getIndexes());
				else
					join.addRight(onRight.apply((T) tmp.get()), tmp.get().getIndexes());
			}
		}, () -> {
			join.close();
			qOut.closeInput();
			future.complete(null);
		});

		if (explicitLeftCaching)
			leftCache.scheduleDeletion();
		if (explicitRightCaching)
			rightCache.scheduleDeletion();

		return future;
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer, Runnable finalizer) {
		return submitOOCTasks(queues, consumer, finalizer, null);
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer, Runnable finalizer, BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed) {
		List<CompletableFuture<Void>> futures = new ArrayList<>(queues.size());

		for (int i = 0; i < queues.size(); i++)
			futures.add(new CompletableFuture<>());

		return submitOOCTasks(queues, consumer, finalizer, futures, null, onNotProcessed);
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer, Runnable finalizer, List<CompletableFuture<Void>> futures, BiFunction<Integer, OOCStream.QueueCallback<T>, Boolean> predicate, BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed) {
		addInStream(queues.toArray(OOCStream[]::new));
		ExecutorService pool = CommonThreadPool.get();

		final List<AtomicInteger> activeTaskCtrs = new ArrayList<>(queues.size());

		for (int i = 0; i < queues.size(); i++)
			activeTaskCtrs.add(new AtomicInteger(1));

		final CompletableFuture<Void> globalFuture = CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
		if (_outQueues == null)
			_outQueues = Collections.emptySet();
		final Runnable oocFinalizer = oocTask(finalizer, null, Stream.concat(_outQueues.stream(), _inQueues.stream()).toArray(OOCStream[]::new));

		int i = 0;
		@SuppressWarnings("unused")
		final int streamId = nextStreamId.getAndIncrement();
		//System.out.println("New stream: (id " + streamId + ", size " + queues.size() + ", initiator '" + this.getClass().getSimpleName() + "')");

		for (OOCStream<T> queue : queues) {
			final int k = i;
			final AtomicInteger localTaskCtr =  activeTaskCtrs.get(k);
			final CompletableFuture<Void> localFuture = futures.get(k);
			final AtomicBoolean closeRaceWatchdog = new AtomicBoolean(false);

			//System.out.println("Substream (k " + k + ", id " + streamId + ", type '" + queue.getClass().getSimpleName() + "', stream_id " + queue.hashCode() + ")");
			queue.setSubscriber(oocTask(callback -> {
				long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
				try (callback) {
					if(callback.isEos()) {
						if(!closeRaceWatchdog.compareAndSet(false, true))
							throw new DMLRuntimeException(
								"Race condition observed: NO_MORE_TASKS callback has been triggered more than once");

						if(localTaskCtr.decrementAndGet() == 0) {
							// Then we can run the finalization procedure already
							localFuture.complete(null);
						}
						return;
					}

					if(predicate != null && !predicate.apply(k, callback)) { // Can get closed due to cancellation
						if(onNotProcessed != null)
							onNotProcessed.accept(k, callback);
						return;
					}

					if(localFuture.isDone()) {
						if(onNotProcessed != null)
							onNotProcessed.accept(k, callback);
						return;
					}
					else {
						localTaskCtr.incrementAndGet();
					}

					// The item needs to be pinned in memory to be accessible in the executor thread
					final OOCStream.QueueCallback<T> pinned = callback.keepOpen();

					pool.submit(oocTask(() -> {
						long taskStartTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
						try (pinned) {
							consumer.accept(k, pinned);

							if(localTaskCtr.decrementAndGet() == 0)
								localFuture.complete(null);
						} finally {
							if (DMLScript.STATISTICS) {
								_localStatisticsAdder.add(System.nanoTime() - taskStartTime);
								if (globalFuture.isDone()) {
									Statistics.maintainOOCHeavyHitter(getExtendedOpcode(), _localStatisticsAdder.sum());
									_localStatisticsAdder.reset();
								}
								if (DMLScript.OOC_LOG_EVENTS)
									OOCEventLog.onComputeEvent(_callerId, taskStartTime, System.nanoTime());
							}
						}
					}, localFuture, Stream.concat(_outQueues.stream(), _inQueues.stream()).toArray(OOCStream[]::new)));

					if(closeRaceWatchdog.get()) // Sanity check
						throw new DMLRuntimeException("Race condition observed");
				} finally {
					if (DMLScript.STATISTICS) {
						_localStatisticsAdder.add(System.nanoTime() - startTime);
						if (globalFuture.isDone()) {
							Statistics.maintainOOCHeavyHitter(getExtendedOpcode(), _localStatisticsAdder.sum());
							_localStatisticsAdder.reset();
						}
					}
				}
			}, null,  Stream.concat(_outQueues.stream(), _inQueues.stream()).toArray(OOCStream[]::new)));

			i++;
		}

		globalFuture.whenComplete((res, e) -> {
			if (globalFuture.isCancelled() || globalFuture.isCompletedExceptionally()) {
				futures.forEach(f -> {
					if(!f.isDone()) {
						if(globalFuture.isCancelled() || globalFuture.isCompletedExceptionally())
							f.cancel(true);
						else
							f.complete(null);
					}
				});
			}

			oocFinalizer.run();
		});
		return globalFuture;
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue, Consumer<OOCStream.QueueCallback<T>> consumer, Runnable finalizer) {
		return submitOOCTasks(List.of(queue), (i, tmp) -> consumer.accept(tmp), finalizer);
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue, Consumer<OOCStream.QueueCallback<T>> consumer, Runnable finalizer, Function<OOCStream.QueueCallback<T>, Boolean> predicate, BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed) {
		return submitOOCTasks(List.of(queue), (i, tmp) -> consumer.accept(tmp), finalizer, List.of(new CompletableFuture<Void>()), (i, tmp) -> predicate.apply(tmp), onNotProcessed);
	}

	protected CompletableFuture<Void> submitOOCTask(Runnable r, OOCStream<?>... queues) {
		ExecutorService pool = CommonThreadPool.get();
		final CompletableFuture<Void> future = new CompletableFuture<>();
		try {
			pool.submit(oocTask(() -> {
				long startTime = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
				r.run();
				future.complete(null);
				if (DMLScript.STATISTICS)
					Statistics.maintainOOCHeavyHitter(getExtendedOpcode(), System.nanoTime() - startTime);
				if (DMLScript.OOC_LOG_EVENTS)
					OOCEventLog.onComputeEvent(_callerId, startTime,  System.nanoTime());
				}, future, queues));
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}

		return future;
	}

	private Runnable oocTask(Runnable r, CompletableFuture<Void> future,  OOCStream<?>... queues) {
		return () -> {
			long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
			try {
				r.run();
			}
			catch (Exception ex) {
				DMLRuntimeException re = ex instanceof DMLRuntimeException ? (DMLRuntimeException) ex : new DMLRuntimeException(ex);

				if (_failed) // Do avoid infinite cycles
					throw re;

				_failed = true;

				for (OOCStream<?> q : queues)
					q.propagateFailure(re);

				if (future != null)
					future.completeExceptionally(re);

				// Rethrow to ensure proper future handling
				throw re;
			} finally {
				if (DMLScript.STATISTICS)
					_localStatisticsAdder.add(System.nanoTime() - startTime);
			}
		};
	}

	private <T> Consumer<OOCStream.QueueCallback<T>> oocTask(Consumer<OOCStream.QueueCallback<T>> c, CompletableFuture<Void> future,  OOCStream<?>... queues) {
		return callback -> {
			try {
				c.accept(callback);
			}
			catch (Exception ex) {
				DMLRuntimeException re = ex instanceof DMLRuntimeException ? (DMLRuntimeException) ex : new DMLRuntimeException(ex);

				if (_failed) // Do avoid infinite cycles
					throw re;

				_failed = true;

				for (OOCStream<?> q : queues)
					q.propagateFailure(re);

				if (future != null)
					future.completeExceptionally(re);

				// Rethrow to ensure proper future handling
				throw re;
			}
		};
	}

	/**
	 * Tracks blocks and their counts to enable early emission
	 * once all blocks for a given index are processed.
	 */
	public static class OOCMatrixBlockTracker {
		private final long _emitThreshold;
		private final HashMap<Long, MatrixBlock> _blocks;
		private final HashMap<Long, Integer> _cnts;

		public OOCMatrixBlockTracker(long emitThreshold) {
			_emitThreshold = emitThreshold;
			_blocks = new HashMap<>();
			_cnts = new HashMap<>();
		}

		/**
		 * Adds or updates a block for the given index and updates its internal count.
		 * @param idx   block index
		 * @param block MatrixBlock
		 * @return true if the block count reached the threshold (ready to emit), false otherwise
		 */
		public boolean putAndIncrementCount(Long idx, MatrixBlock block) {
			_blocks.put(idx, block);
			int newCnt = _cnts.getOrDefault(idx, 0) + 1;
			if( newCnt < _emitThreshold )
				_cnts.put(idx, newCnt);
			return newCnt == _emitThreshold;
		}

		public boolean incrementCount(Long idx) {
			int newCnt = _cnts.get(idx) + 1;
			if( newCnt < _emitThreshold )
				_cnts.put(idx, newCnt);
			return newCnt == _emitThreshold;
		}

		public void putAndInitCount(Long idx, MatrixBlock block) {
			_blocks.put(idx, block);
			_cnts.put(idx, 0);
		}

		public MatrixBlock get(Long idx) {
			return _blocks.get(idx);
		}

		public void remove(Long idx) {
			_blocks.remove(idx);
			_cnts.remove(idx);
		}
	}
}
