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
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.runtime.ooc.stream.FilteredOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.stream.TaskContext;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.Statistics;
import scala.Tuple2;
import scala.Tuple4;
import scala.Tuple5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public abstract class OOCInstruction extends Instruction {
	public static final ExecutorService COMPUTE_EXECUTOR = CommonThreadPool.get();
	private static final AtomicInteger COMPUTE_IN_FLIGHT = new AtomicInteger(0);
	private static final int COMPUTE_BACKPRESSURE_THRESHOLD = 100;
	protected static final Log LOG = LogFactory.getLog(OOCInstruction.class.getName());
	private static final AtomicInteger nextStreamId = new AtomicInteger(0);
	private long nanoTime;

	public enum OOCType {
		Reblock, Tee, Binary, Ternary, Unary, AggregateUnary, AggregateBinary, AggregateTernary, MAPMM, MMTSJ,
		MAPMMCHAIN, Reorg, CM, Ctable, MatrixIndexing, ParameterizedBuiltin, Rand
	}

	protected final OOCInstruction.OOCType _ooctype;
	protected final boolean _requiresLabelUpdate;
	protected StreamContext _streamContext;
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

		if (DMLScript.STATISTICS)
			_localStatisticsAdder = new LongAdder();
		_callerId = DMLScript.OOC_LOG_EVENTS ? OOCEventLog.registerCaller(getExtendedOpcode() + "_" + hashCode()) : 0;
	}

	public static int getComputeInFlight() {
		return COMPUTE_IN_FLIGHT.get();
	}

	public static int getComputeBackpressureThreshold() {
		return COMPUTE_BACKPRESSURE_THRESHOLD;
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
		_streamContext = null;
		if(DMLScript.LINEAGE_DEBUGGER)
			ec.maintainLineageDebuggerInfo(this);
		if (DMLScript.OOC_LOG_EVENTS)
			OOCEventLog.onComputeEvent(_callerId, nanoTime, System.nanoTime());
	}

	protected void addInStream(OOCStream<?>... queue) {
		if(_streamContext == null)
			_streamContext = new StreamContext();
		_streamContext.addInStream(queue);
	}

	protected void addOutStream(OOCStream<?>... queue) {
		if(_streamContext == null)
			_streamContext = new StreamContext();
		_streamContext.addOutStream(queue);
	}

	protected boolean inStreamsDefined() {
		return _streamContext != null && _streamContext.inStreamsDefined();
	}

	protected boolean outStreamsDefined() {
		return _streamContext != null && _streamContext.outStreamsDefined();
	}

	protected <T> OOCStream<T> createWritableStream() {
		return new SubscribableTaskQueue<>();
	}

	protected <T> CompletableFuture<Void> filterOOC(OOCStream<T> qIn, Consumer<T> processor, Function<T, Boolean> predicate) {
		return filterOOC(qIn, processor, predicate, null);
	}

	protected <T> CompletableFuture<Void> filterOOC(OOCStream<T> qIn, Consumer<T> processor, Function<T, Boolean> predicate, Consumer<T> onNotProcessed) {
		if (!inStreamsDefined() || !outStreamsDefined())
			throw new NotImplementedException("filterOOC requires manual specification of all input and output streams for error propagation");

		return submitOOCTasks(qIn, c -> processor.accept(c.get()), p -> predicate.apply(p.get()), onNotProcessed != null ? (i, tmp) -> onNotProcessed.accept(tmp.get()) : null);
	}

	protected <T> OOCStream<T> filteredOOCStream(OOCStream<T> qIn, Function<T, Boolean> predicate) {
		return new FilteredOOCStream<>(qIn, predicate);
	}

	protected <T, R> CompletableFuture<Void> mapOOC(OOCStream<T> qIn, OOCStream<R> qOut, Function<T, R> mapper) {
		return mapOptionalOOC(qIn, qOut, tmp -> Optional.of(mapper.apply(tmp)));
	}

	protected <T, R> CompletableFuture<Void> mapOptionalOOC(OOCStream<T> qIn, OOCStream<R> qOut, Function<T, Optional<R>> optionalMapper) {
		addInStream(qIn);
		addOutStream(qOut);

		AtomicInteger deferredCtr = new AtomicInteger(1);
		CompletableFuture<Void> future = new CompletableFuture<>();

		Consumer<OOCStream.QueueCallback<T>> exec = tmp -> {
			Optional<R> r;
			try(tmp) {
				r = optionalMapper.apply(tmp.get());
			}
			catch(Exception e) {
				throw e instanceof DMLRuntimeException ? (DMLRuntimeException) e : new DMLRuntimeException(e);
			}
			r.ifPresent(t -> {
				deferredCtr.incrementAndGet();
				// Defer to clean the stack of large objects
				TaskContext.defer(() -> {
					qOut.enqueue(t);
					if(deferredCtr.decrementAndGet() == 0)
						future.complete(null);
				});
			});
		};

		submitOOCTasks(qIn, exec, tmp -> {
					// Try to run as a predicate to prefer pipelining rather than fan-out
					if(ForkJoinTask.getPool() == COMPUTE_EXECUTOR) {
						exec.accept(tmp);
						return false;
					}
					return true;
				}, null)
			.thenRun(() -> {
				if(deferredCtr.decrementAndGet() == 0)
					future.complete(null);
			})
			.exceptionally(err -> {
				future.completeExceptionally(err);
				return null;
			});

		return future.thenRun(qOut::closeInput).exceptionally(err -> {
			DMLRuntimeException dmlErr = DMLRuntimeException.of(err);
			qOut.propagateFailure(dmlErr);
			throw dmlErr;
		});
	}

	protected <R, P> CompletableFuture<Void> broadcastJoinOOC(OOCStream<IndexedMatrixValue> qIn, OOCStream<IndexedMatrixValue> broadcast, OOCStream<R> qOut, BiFunction<IndexedMatrixValue, BroadcastedElement, R> mapper, Function<IndexedMatrixValue, P> on) {
		return broadcastJoinOOC(qIn, broadcast, qOut, mapper, on, on);
	}

	protected <R, P> CompletableFuture<Void> broadcastJoinOOC(OOCStream<IndexedMatrixValue> qIn, OOCStream<IndexedMatrixValue> broadcast, OOCStream<R> qOut, BiFunction<IndexedMatrixValue, BroadcastedElement, R> mapper, Function<IndexedMatrixValue, P> onLeft, Function<IndexedMatrixValue, P> onRight) {
		addInStream(qIn, broadcast);
		addOutStream(qOut);

		CachingStream leftCache = qIn.hasStreamCache() ? qIn.getStreamCache() : new CachingStream(qIn);
		CachingStream rightCache = broadcast.hasStreamCache() ? broadcast.getStreamCache() : new CachingStream(broadcast);
		leftCache.activateIndexing();
		rightCache.activateIndexing();

		leftCache.incrSubscriberCount(1); // Prevent early block deletion as we may read elements twice
		rightCache.incrSubscriberCount(1);

		Map<P, List<MatrixIndexes>> availableLeftInput = new ConcurrentHashMap<>();
		Map<P, BroadcastedElement> availableBroadcastInput = new ConcurrentHashMap<>();

		OOCStream<Tuple4<P, OOCStream.QueueCallback<IndexedMatrixValue>, OOCStream.QueueCallback<IndexedMatrixValue>, BroadcastedElement>> broadcastingQueue = createWritableStream();
		AtomicInteger waitCtr = new AtomicInteger(1);
		Object lock = new Object();

		CompletableFuture<Void> fut1 = submitOOCTasks(List.of(leftCache.getReadStream(), rightCache.getReadStream()), (i, tmp) -> {
			try(tmp) {
				P key = i == 0 ? onLeft.apply(tmp.get()) : onRight.apply(tmp.get());

				if(i == 0) { // qIn stream
					BroadcastedElement b;

					synchronized(lock) {
						b = availableBroadcastInput.get(key);

						if(b == null) {
							availableLeftInput.compute(key, (k, v) -> {
								if(v == null)
									v = new ArrayList<>();
								v.add(tmp.get().getIndexes());
								return v;
							});
							return;
						}
					}

					// Then items are present in cache
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
				else { // broadcast stream
					BroadcastedElement b = new BroadcastedElement(tmp.get().getIndexes());
					List<MatrixIndexes> queued;
					synchronized(lock) {
						availableBroadcastInput.put(key, b);
						queued = availableLeftInput.remove(key);
					}

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
		});
		fut1 = fut1.thenApply(v -> {
			if(waitCtr.decrementAndGet() == 0)
				broadcastingQueue.closeInput();
			return null;
		});

		CompletableFuture<Void> fut2 = submitOOCTasks(List.of(broadcastingQueue), (i, tpl) -> {
			try(tpl) {
				final BroadcastedElement b = tpl.get()._4();
				final OOCStream.QueueCallback<IndexedMatrixValue> lValue = tpl.get()._2();
				final OOCStream.QueueCallback<IndexedMatrixValue> bValue = tpl.get()._3();

				try(lValue; bValue) {
					b.value = bValue.get();
					leftCache.incrProcessingCount(leftCache.findCachedIndex(lValue.get().getIndexes()), 1);
					qOut.enqueue(mapper.apply(lValue.get(), b));

					if(b.tryRelease()) {
						availableBroadcastInput.remove(tpl.get()._1());
						rightCache.incrProcessingCount(rightCache.findCachedIndex(b.idx), 1); // Correct for incremented subscriber count to allow block deletion
					}
				}

				if(waitCtr.decrementAndGet() == 0)
					broadcastingQueue.closeInput();
			}
		});

		if(!qIn.hasStreamCache())
			leftCache.scheduleDeletion();
		if(!broadcast.hasStreamCache())
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

	protected <R, P> CompletableFuture<Void> joinManyOOC(OOCStream<IndexedMatrixValue> left,
		OOCStream<IndexedMatrixValue> right, OOCStream<R> out,
		BiFunction<IndexedMatrixValue, IndexedMatrixValue, R> mapper, Function<IndexedMatrixValue, P> leftOn,
		Function<IndexedMatrixValue, P> rightOn, int releaseLeftCount, int releaseRightCount) {
		addInStream(left, right);
		addOutStream(out);

		CachingStream leftCache = left.hasStreamCache() ? left.getStreamCache() : new CachingStream(left);
		CachingStream rightCache = right.hasStreamCache() ? right.getStreamCache() : new CachingStream(right);
		leftCache.activateIndexing();
		rightCache.activateIndexing();

		leftCache.incrSubscriberCount(1); // Prevent early block deletion as we may read elements twice
		rightCache.incrSubscriberCount(1);

		Map<P, Tuple2<List<BroadcastedElement>, List<BroadcastedElement>>> joinMap = new ConcurrentHashMap<>();

		OOCStream<Tuple5<P, OOCStream.QueueCallback<IndexedMatrixValue>, OOCStream.QueueCallback<IndexedMatrixValue>, BroadcastedElement, BroadcastedElement>> joinQueue = createWritableStream();
		AtomicInteger waitCtr = new AtomicInteger(1);

		CompletableFuture<Void> fut1 = submitOOCTasks(List.of(leftCache.getReadStream(), rightCache.getReadStream()),
			(i, tmp) -> {
				try(tmp) {
					boolean leftItem = i == 0;
					P key = (leftItem ? leftOn : rightOn).apply(tmp.get());
					Tuple2<List<BroadcastedElement>, List<BroadcastedElement>> tuple = joinMap.computeIfAbsent(key,
						k -> new Tuple2<>(new ArrayList<>(releaseRightCount), new ArrayList<>(releaseLeftCount)));
					BroadcastedElement b = new BroadcastedElement(tmp.get().getIndexes());
					List<BroadcastedElement> matches = leftItem ? tuple._2 : tuple._1;
					List<BroadcastedElement> toInsert = leftItem ? tuple._1 : tuple._2;
					boolean remove;
					synchronized(tuple) {
						toInsert.add(b);

						for(BroadcastedElement e : matches) {
							waitCtr.incrementAndGet();
							OOCCacheManager.requestManyBlocks(
								List.of(leftCache.peekCachedBlockKey(leftItem ? b.idx : e.idx),
									rightCache.peekCachedBlockKey(leftItem ? e.idx : b.idx))).thenApply(joined -> {
								try {
									joinQueue.enqueue(
										new Tuple5<>(key, joined.get(0).keepOpen(), joined.get(1).keepOpen(),
											leftItem ? b : e, leftItem ? e : b));
								}
								finally {
									joined.forEach(OOCStream.QueueCallback::close);
								}
								return null;
							}).exceptionally(t -> {
								joinQueue.propagateFailure(DMLRuntimeException.of(t));
								return null;
							});
						}
						remove = tuple._1.size() == releaseRightCount && tuple._2.size() == releaseLeftCount;
					}
					if(remove)
						joinMap.remove(key);
				}
			});
		fut1 = fut1.thenApply(v -> {
			if(waitCtr.decrementAndGet() == 0)
				joinQueue.closeInput();
			return null;
		});

		CompletableFuture<Void> fut2 = mapOOC(joinQueue, out, tpl -> {
			final BroadcastedElement bLeft = tpl._4();
			final BroadcastedElement bRight = tpl._5();
			final OOCStream.QueueCallback<IndexedMatrixValue> lValue = tpl._2();
			final OOCStream.QueueCallback<IndexedMatrixValue> rValue = tpl._3();
			R res;

			try(lValue; rValue) {
				res = mapper.apply(lValue.get(), rValue.get());
				int leftCtr = bLeft.incrProcessCtrAndGet();
				int rightCtr = bRight.incrProcessCtrAndGet();

				if(leftCtr == releaseLeftCount)
					leftCache.incrProcessingCount(leftCache.findCachedIndex(bLeft.idx),
						1); // Correct for incremented subscriber count to allow block deletion
				if(rightCtr == releaseRightCount)
					rightCache.incrProcessingCount(rightCache.findCachedIndex(bRight.idx), 1);
			}

			if(waitCtr.decrementAndGet() == 0)
				joinQueue.closeInput();
			return res;
		});

		if(!left.hasStreamCache())
			leftCache.scheduleDeletion();
		if(!right.hasStreamCache())
			rightCache.scheduleDeletion();

		return CompletableFuture.allOf(fut1, fut2);
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

		public synchronized boolean tryRelease() {
			if(release) {
				release = false; // To not double release
				return true;
			}
			return false;
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

	protected <R> CompletableFuture<Void> joinOOC(OOCStream<IndexedMatrixValue> qIn1, OOCStream<IndexedMatrixValue> qIn2, OOCStream<R> qOut, BiFunction<IndexedMatrixValue, IndexedMatrixValue, R> mapper, Function<IndexedMatrixValue, MatrixIndexes> on) {
		return joinOOC(List.of(qIn1, qIn2), qOut, t -> mapper.apply(t.get(0), t.get(1)), on);
	}

	protected <R> CompletableFuture<Void> joinOOC(List<OOCStream<IndexedMatrixValue>> qIn, OOCStream<R> qOut, Function<List<IndexedMatrixValue>, R> mapper, Function<IndexedMatrixValue, MatrixIndexes> on) {
		int inSize = qIn.size();
		return joinOOC(qIn, qOut, mapper, Collections.nCopies(inSize, on), t -> Collections.nCopies(inSize, t));
	}

	protected <R, P> CompletableFuture<Void> joinOOC(List<OOCStream<IndexedMatrixValue>> qIn, OOCStream<R> qOut, Function<List<IndexedMatrixValue>, R> mapper, List<Function<IndexedMatrixValue, P>> on, Function<P, List<MatrixIndexes>> invOn) {
		if(qIn == null || on == null || qIn.size() != on.size())
			throw new DMLRuntimeException("joinOOC(list) requires the same number of streams and key functions.");

		final int n = qIn.size();

		CachingStream[] caches = new CachingStream[n];
		boolean[] explicitCaching = new boolean[n];

		for(int i = 0; i < n; i++) {
			OOCStream<IndexedMatrixValue> s = qIn.get(i);
			explicitCaching[i] = !s.hasStreamCache();
			caches[i] = explicitCaching[i] ? new CachingStream(s) : s.getStreamCache();
			caches[i].activateIndexing();
			// One additional consumption for the materialization when emitting
			caches[i].incrSubscriberCount(1);
		}

		Map<P, MatrixIndexes[]> seen = new ConcurrentHashMap<>();

		OOCStream<List<OOCStream.QueueCallback<IndexedMatrixValue>>> materialized = createWritableStream();

		List<OOCStream<IndexedMatrixValue>> rStreams = new ArrayList<>(caches.length);
		for(int i = 0; i < caches.length; i++)
			rStreams.add(explicitCaching[i] ? caches[i].getReadStream() : qIn.get(i));

		AtomicInteger processing = new AtomicInteger(1);

		addInStream(qIn.toArray(OOCStream[]::new));
		addOutStream(qOut);

		CompletableFuture<Void> future = pipeOOC(rStreams, (i, tmp) -> {
			Function<IndexedMatrixValue, P> keyFn = on.get(i);
			P key = keyFn.apply(tmp.get());
			MatrixIndexes idx = tmp.get().getIndexes();

			MatrixIndexes[] arr = seen.computeIfAbsent(key, k -> new MatrixIndexes[n]);
			boolean ready;
			synchronized(arr) {
				arr[i] = idx;
				ready = true;
				for(int j = 0; j < arr.length; j++) {
					MatrixIndexes ix = arr[j];
					if (ix == null) {
						ready = false;
						break;
					}
				}
			}

			if(!ready || !seen.remove(key, arr))
				return;

			processing.incrementAndGet();
			List<BlockKey> entries = new ArrayList<>(arr.length);
			for(int j = 0; j < arr.length; j++)
				entries.add(caches[j].peekCachedBlockKey(arr[j]));

			var f = OOCCacheManager.requestManyBlocks(entries);
			f.whenComplete((r, err) -> {
				try {
					if(err != null) {
						if(err instanceof DMLRuntimeException)
							materialized.propagateFailure((DMLRuntimeException) err);
						else if(err instanceof Exception)
							materialized.propagateFailure(new DMLRuntimeException(err));
						else
							materialized.propagateFailure(new DMLRuntimeException(new Exception(err)));
						return;
					}
					List<OOCStream.QueueCallback<IndexedMatrixValue>> outList = new ArrayList<>(r.size());
					for(int j = 0; j < r.size(); j++) {
						if(explicitCaching[j]) {
							// Early forget item from cache
							outList.add(new OOCStream.SimpleQueueCallback<>(r.get(j).get(), null));
						}
						else {
							outList.add(r.get(j).keepOpen());
						}
						caches[j].incrProcessingCount(caches[j].findCachedIndex(r.get(j).get().getIndexes()), 1);
					}
					materialized.enqueue(outList);
					r.forEach(OOCStream.QueueCallback::close);
				}
				catch(Throwable t) {
					throw t;
				}
				finally {
					if(processing.decrementAndGet() == 0)
						materialized.closeInput();
				}
			});
		});

		future.whenComplete((r, err) -> {
			if (processing.decrementAndGet() == 0) {
				materialized.closeInput();
			}
		});

		CompletableFuture<Void> outFuture = mapOOC(materialized, qOut, cb -> {
			try {
				List<IndexedMatrixValue> imvs = cb.stream().map(OOCStream.QueueCallback::get).toList();
				return mapper.apply(imvs);
			}
			finally {
				cb.forEach(OOCStream.QueueCallback::close);
			}
		});

		for(int i = 0; i < n; i++) {
			if (explicitCaching[i])
				caches[i].scheduleDeletion();
		}

		return outFuture;
	}

	protected CompletableFuture<Void> groupedReduceOOC(OOCStream<IndexedMatrixValue> qIn, OOCStream<IndexedMatrixValue> qOut, BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> reduce, int emitCount) {
		addInStream(qIn);
		addOutStream(qOut);

		if(qIn.hasStreamCache())
			throw new UnsupportedOperationException();
		Map<MatrixIndexes, Aggregator> aggregators = new ConcurrentHashMap<>();
		AtomicInteger busyCtr = new AtomicInteger(1);
		CompletableFuture<Void> outFuture = new CompletableFuture<>();

		CompletableFuture<Void> pipeFuture = pipeOOC(qIn, cb -> {
			try(cb) {
				Aggregator agg = aggregators.compute(cb.get().getIndexes(), (k, v) -> {
					if(v == null) {
						v = new Aggregator(reduce, emitCount);
						busyCtr.incrementAndGet();
						v.getFuture().thenApply(imv -> {
							qOut.enqueue(imv);
							if(busyCtr.decrementAndGet() == 0)
								outFuture.complete(null);
							return null;
						})
							.exceptionally(outFuture::completeExceptionally);
					}
					return v;
				});
				agg.insert(cb.get());
			}
		});

		pipeFuture.thenRun(() -> {
			if(busyCtr.decrementAndGet() == 0)
				outFuture.complete(null);
		});

		return outFuture.thenRun(qOut::closeInput);
	}

	private static class Aggregator {
		private final long _streamId;
		private final BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> _aggFn;
		private final int _numTiles;
		private final CompletableFuture<IndexedMatrixValue> _future;
		private LinkedList<BlockKey> _availableIntermediates;
		private int _blockId;
		private int _processed;


		public Aggregator(BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> aggFn, int numTiles) {
			_streamId = CachingStream._streamSeq.getNextID();
			_blockId = 0;
			_aggFn = aggFn;
			_numTiles = numTiles;
			_future = new CompletableFuture<>();
			_availableIntermediates = new LinkedList<>();
			_processed = 0;
		}

		public CompletableFuture<IndexedMatrixValue> getFuture() {
			return _future;
		}

		public void insert(IndexedMatrixValue imv) {
			IndexedMatrixValue v = null;
			CompletableFuture<List<BlockEntry>> future = null;
			boolean finished = false;
			synchronized(this) {
				_processed++;
				if(_processed == _numTiles * 2 - 1) {
					// Then we are done
					finished = true;
				}
				else {
					if(!_availableIntermediates.isEmpty()) {
						List<BlockKey> sel = new ArrayList<>(1);
						List<BlockEntry> entries = OOCCacheManager.getCache().tryRequestAnyOf(_availableIntermediates, 1, sel);

						if(entries == null) {
							BlockEntry entry = OOCCacheManager.getCache()
								.putAndPin(new BlockKey(_streamId, _blockId++), imv,
									((MatrixBlock) imv.getValue()).getExactSerializedSize());
							entry.addRetainHint(10);
							future = OOCCacheManager.getCache()
								.request(List.of(entry.getKey(), _availableIntermediates.removeFirst()));
							OOCCacheManager.getCache().unpin(entry);
						}
						else {
							v = (IndexedMatrixValue)entries.get(0).getData();
							_availableIntermediates.remove(sel.get(0));
							OOCCacheManager.getCache().forget(sel.get(0));
							if(v == null)
								throw new IllegalStateException();
						}
					}
					else {
						BlockEntry entry = OOCCacheManager.getCache()
							.putAndPin(new BlockKey(_streamId, _blockId++), imv,
								((MatrixBlock) imv.getValue()).getExactSerializedSize());
						entry.addRetainHint(10);
						OOCCacheManager.getCache().unpin(entry);
						_availableIntermediates.add(entry.getKey());
						return;
					}
				}
			}

			if(finished) {
				_availableIntermediates = null;
				_future.complete(imv);
				return;
			}

			if(v != null) {
				imv = _aggFn.apply(v, imv);
				insert(imv);
				return;
			}

			future.thenApply(l -> {
				IndexedMatrixValue agg = _aggFn.apply((IndexedMatrixValue)l.get(0).getData(),
					(IndexedMatrixValue)l.get(1).getData());
				OOCCacheManager.getCache().forget(l.get(0).getKey());
				OOCCacheManager.getCache().forget(l.get(1).getKey());
				insert(agg);
				return null;
			});
		}
	}

	protected <T> CompletableFuture<Void> pipeOOC(OOCStream<T> queue, Consumer<OOCStream.QueueCallback<T>> consumer) {
		return pipeOOC(List.of(queue), (i, tmp) -> consumer.accept(tmp));
	}

	protected <T> CompletableFuture<Void> pipeOOC(List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer) {
		return submitOOCTasks(queues, consumer, (i, tmp) -> {
			// Try to run as a predicate to prefer pipelining rather than fan-out
			if(ForkJoinTask.getPool() == COMPUTE_EXECUTOR) {
				consumer.accept(i, tmp);
				return false;
			}
			return true;
		}, (i, tmp) -> {});
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer) {
		return submitOOCTasks(queues, consumer, null, null);
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, OOCStream.QueueCallback<T>> consumer, BiFunction<Integer, OOCStream.QueueCallback<T>, Boolean> predicate, BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed) {
		addInStream(queues.toArray(OOCStream[]::new));
		if(!outStreamsDefined())
			throw new IllegalArgumentException("Explicit specification of all output streams is required before submitting tasks. If no output streams are present use addOutStream().");

		final List<AtomicInteger> activeTaskCtrs = new ArrayList<>(queues.size());
		final List<CompletableFuture<Void>> futures = new ArrayList<>(queues.size());

		for(int i = 0; i < queues.size(); i++) {
			activeTaskCtrs.add(new AtomicInteger(1));
			futures.add(new CompletableFuture<>());
		}

		final CompletableFuture<Void> globalFuture = CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
		final StreamContext streamContext = _streamContext.copy(); // Snapshot of the current stream context
		if(streamContext == null || !streamContext.inStreamsDefined() || !streamContext.outStreamsDefined())
			throw new IllegalArgumentException("Explicit specification of all output streams is required before submitting tasks. If no output streams are present use addOutStream().");

		int i = 0;
		@SuppressWarnings("unused")
		final int streamId = nextStreamId.getAndIncrement();

		for (OOCStream<T> queue : queues) {
			final int k = i;
			final AtomicInteger localTaskCtr =  activeTaskCtrs.get(k);
			final CompletableFuture<Void> localFuture = futures.get(k);
			final AtomicBoolean closeRaceWatchdog = new AtomicBoolean(false);

			queue.setSubscriber(oocTask(callback -> {
				long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
				try(callback) {
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

					COMPUTE_IN_FLIGHT.incrementAndGet();
					try {
						Runnable oocTask = oocTask(() -> {
							long taskStartTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
							try(pinned) {
								consumer.accept(k, pinned);

								if(localTaskCtr.decrementAndGet() == 0) {
									TaskContext.defer(() -> localFuture.complete(null));
								}
							}
							finally {
								COMPUTE_IN_FLIGHT.decrementAndGet();
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
						}, localFuture, streamContext);
						COMPUTE_EXECUTOR.submit(oocTask);
					}
					catch (Exception e) {
						COMPUTE_IN_FLIGHT.decrementAndGet();
						throw e;
					}

					if(closeRaceWatchdog.get()) // Sanity check
						throw new DMLRuntimeException("Race condition observed");
				}
				catch(Throwable t) {
					streamContext.failAll(DMLRuntimeException.of(t));
					throw t;
				}
				finally {
					if (DMLScript.STATISTICS) {
						_localStatisticsAdder.add(System.nanoTime() - startTime);
						if (globalFuture.isDone()) {
							Statistics.maintainOOCHeavyHitter(getExtendedOpcode(), _localStatisticsAdder.sum());
							_localStatisticsAdder.reset();
						}
					}
				}
			}, null,  streamContext));

			i++;
		}

		return globalFuture.handle((res, e) -> {
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

			streamContext.clear();
			return null;
		});
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue, Consumer<OOCStream.QueueCallback<T>> consumer) {
		return submitOOCTasks(List.of(queue), (i, tmp) -> consumer.accept(tmp), null, null);
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(OOCStream<T> queue, Consumer<OOCStream.QueueCallback<T>> consumer, Function<OOCStream.QueueCallback<T>, Boolean> predicate, BiConsumer<Integer, OOCStream.QueueCallback<T>> onNotProcessed) {
		return submitOOCTasks(List.of(queue), (i, tmp) -> consumer.accept(tmp), (i, tmp) -> predicate.apply(tmp), onNotProcessed);
	}

	protected CompletableFuture<Void> submitOOCTask(Runnable r, StreamContext ctx) {
		ExecutorService pool = CommonThreadPool.get();
		final CompletableFuture<Void> future = new CompletableFuture<>();
		try {
			COMPUTE_IN_FLIGHT.incrementAndGet();
			pool.submit(oocTask(() -> {
				long startTime = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
				try {
					r.run();
					future.complete(null);
					ctx.clear();
					if (DMLScript.STATISTICS)
						Statistics.maintainOOCHeavyHitter(getExtendedOpcode(), System.nanoTime() - startTime);
					if (DMLScript.OOC_LOG_EVENTS)
						OOCEventLog.onComputeEvent(_callerId, startTime,  System.nanoTime());
				}
				finally {
					COMPUTE_IN_FLIGHT.decrementAndGet();
				}
				}, future, ctx));
		}
		catch (Exception ex) {
			COMPUTE_IN_FLIGHT.decrementAndGet();
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}

		return future;
	}

	private Runnable oocTask(Runnable r, CompletableFuture<Void> future,  StreamContext ctx) {
		return () -> {
			boolean setContext = TaskContext.getContext() == null;
			if(setContext)
				TaskContext.setContext(new TaskContext());
			long startTime = DMLScript.STATISTICS ? System.nanoTime() : 0;
			try {
				r.run();
				if(setContext) {
					while(TaskContext.runDeferred()) {
					}
				}
			}
			catch (Exception ex) {
				DMLRuntimeException re = DMLRuntimeException.of(ex);

				ctx.failAll(re);

				if (future != null)
					future.completeExceptionally(re);

				// Rethrow to ensure proper future handling
				throw re;
			} finally {
				if(setContext)
					TaskContext.clearContext();
				if (DMLScript.STATISTICS)
					_localStatisticsAdder.add(System.nanoTime() - startTime);
			}
		};
	}

	private <T> Consumer<OOCStream.QueueCallback<T>> oocTask(Consumer<OOCStream.QueueCallback<T>> c, CompletableFuture<Void> future,  StreamContext ctx) {
		return callback -> {
			try {
				c.accept(callback);
			}
			catch (Exception ex) {
				DMLRuntimeException re = DMLRuntimeException.of(ex);

				ctx.failAll(re);

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
