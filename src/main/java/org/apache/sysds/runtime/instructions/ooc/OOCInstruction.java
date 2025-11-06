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
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.OOCJoin;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public abstract class OOCInstruction extends Instruction {
	protected static final Log LOG = LogFactory.getLog(OOCInstruction.class.getName());
	private static final AtomicInteger nextStreamId = new AtomicInteger(0);

	public enum OOCType {
		Reblock, Tee, Binary, Unary, AggregateUnary, AggregateBinary, MAPMM, MMTSJ, Reorg, CM, Ctable
	}

	protected final OOCInstruction.OOCType _ooctype;
	protected final boolean _requiresLabelUpdate;
	protected Set<OOCStream<?>> _queues;

	protected OOCInstruction(OOCInstruction.OOCType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected OOCInstruction(OOCInstruction.OOCType type, Operator op, String opcode, String istr) {
		super(op);
		_ooctype = type;
		instString = istr;
		instOpcode = opcode;

		_requiresLabelUpdate = super.requiresLabelUpdate();
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
		// TODO
		return super.preprocessInstruction(ec);
	}

	@Override
	public abstract void processInstruction(ExecutionContext ec);

	@Override
	public void postprocessInstruction(ExecutionContext ec) {
		if(DMLScript.LINEAGE_DEBUGGER)
			ec.maintainLineageDebuggerInfo(this);
	}

	protected void addInStream(OOCStream<?>... queue) {
		if (_queues == null)
			_queues = new HashSet<>();
		_queues.addAll(List.of(queue));
	}

	protected void addOutStream(OOCStream<?>... queue) {
		// Currently same behavior as addInQueue
		addInStream(queue);
	}

	protected <T> OOCStream<T> createWritableStream() {
		return new SubscribableTaskQueue<>();
	}

	protected <T, R> void mapOOC(OOCStream<T> qIn, OOCStream<R> qOut, Function<T, R> mapper) {
		addInStream(qIn);
		addOutStream(qOut);

		submitOOCTasks(qIn, tmp -> {
			try {
				R r = mapper.apply(tmp);
				qOut.enqueue(r);
			} catch (Exception e) {
				throw e instanceof DMLRuntimeException ? (DMLRuntimeException) e : new DMLRuntimeException(e);
			}
		}, qOut::closeInput);
	}

	protected <T, R, P> CompletableFuture<Void> joinOOC(OOCStream<T> qIn1, OOCStream<T> qIn2, OOCStream<R> qOut, BiFunction<T, T, R> mapper, Function<T, P> on) {
		return joinOOC(qIn1, qIn2, qOut, mapper, on, on);
	}

	protected <T, R, P> CompletableFuture<Void> joinOOC(OOCStream<T> qIn1, OOCStream<T> qIn2, OOCStream<R> qOut, BiFunction<T, T, R> mapper, Function<T, P> onLeft, Function<T, P> onRight) {
		addInStream(qIn1, qIn2);
		addOutStream(qOut);

		final CompletableFuture<Void> future = new CompletableFuture<>();

		// We need to construct our own stream to properly manage the cached items in the hash join
		CachingStream leftCache = qIn1.hasStreamCache() ? qIn1.getStreamCache() : new CachingStream((SubscribableTaskQueue<IndexedMatrixValue>)qIn1); // We have to assume this generic type for now
		CachingStream rightCache = qIn2.hasStreamCache() ? qIn2.getStreamCache() : new CachingStream((SubscribableTaskQueue<IndexedMatrixValue>)qIn2); // We have to assume this generic type for now
		leftCache.activateIndexing();
		rightCache.activateIndexing();

		final OOCJoin<P, MatrixIndexes> join = new OOCJoin<>((idx, left, right) -> {
			T leftObj = (T) leftCache.findCached(left);
			T rightObj = (T) rightCache.findCached(right);
			qOut.enqueue(mapper.apply(leftObj, rightObj));
		});

		submitOOCTasks(List.of(leftCache.getReadStream(), rightCache.getReadStream()), (i, tmp) -> {
			if (i == 0)
				join.addLeft(onLeft.apply((T)tmp), ((IndexedMatrixValue) tmp).getIndexes());
			else
				join.addRight(onRight.apply((T)tmp), ((IndexedMatrixValue) tmp).getIndexes());
		}, () -> {
			join.close();
			qOut.closeInput();
			future.complete(null);
		});

		return future;
	}

	protected <T> CompletableFuture<Void> submitOOCTasks(final List<OOCStream<T>> queues, BiConsumer<Integer, T> consumer, Runnable finalizer) {
		addInStream(queues.toArray(OOCStream[]::new));
		ExecutorService pool = CommonThreadPool.get();
		final AtomicInteger activeTaskCtr = new AtomicInteger(0);

		final Object lock = new Object();

		List<CompletableFuture<Void>> futures = new ArrayList<>(queues.size());
		final List<AtomicBoolean> streamsClosed = new ArrayList<>(queues.size());
		for (int i = 0; i < queues.size(); i++) {
			streamsClosed.add(new AtomicBoolean(false));
		}

		int i = 0;
		final int streamId = nextStreamId.getAndIncrement();
		//System.out.println("New stream: (id " + streamId + ", size " + queues.size() + ", initiator '" + this.getClass().getSimpleName() + "')");

		for (OOCStream<T> queue : queues) {
			final int k = i;
			final CompletableFuture<Void> localFuture = new CompletableFuture<>();
			final AtomicBoolean localStreamClosed = streamsClosed.get(k);
			futures.add(localFuture);
			//System.out.println("Substream (k " + k + ", id " + streamId + ", type '" + queue.getClass().getSimpleName() + "', stream_id " + queue.hashCode() + ")");
			queue.setSubscriber(() -> {
				try {
					activeTaskCtr.incrementAndGet();
					pool.submit(oocTask(() -> {
						try {
							T item = queue.dequeue();
							if(item != null) {
								//System.out.println("Accept" + ((IndexedMatrixValue)item).getIndexes() + " (k " + k + ", id " + streamId + ")");
								consumer.accept(k, item);
							} else {
								//System.out.println("Close substream (k " + k + ", id " + streamId + ")");
								localStreamClosed.set(true);
							}
							activeTaskCtr.decrementAndGet();

							boolean shutdown;
							synchronized(lock) {
								shutdown = activeTaskCtr.get() == 0 && streamsClosed.stream().allMatch(AtomicBoolean::get);
							}

							if(shutdown) {
								//System.out.println("Shutdown (id " + streamId + ")");
								finalizer.run();
							}
						}
						catch(Exception e) {
							throw (e instanceof DMLRuntimeException ? (DMLRuntimeException)e : new DMLRuntimeException(e));
						}
					}, localFuture, _queues.toArray(OOCStream[]::new)));
				} catch (Exception e) {
					throw new DMLRuntimeException(e);
				}
			});

			i++;
		}

		pool.shutdown();
		return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
	}

	protected <T> void submitOOCTasks(OOCStream<T> queue, Consumer<T> consumer, Runnable finalizer) {
		submitOOCTasks(List.of(queue), (i, tmp) -> consumer.accept(tmp), finalizer);
	}

	protected CompletableFuture<Void> submitOOCTask(Runnable r, OOCStream<?>... queues) {
		ExecutorService pool = CommonThreadPool.get();
		final CompletableFuture<Void> future = new CompletableFuture<>();
		try {
			pool.submit(oocTask(() -> {r.run();future.complete(null);}, future, queues));
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}

		return future;
	}

	private Runnable oocTask(Runnable r, CompletableFuture<Void> future, OOCStream<?>... queues) {
		return () -> {
			try {
				r.run();
			}
			catch (Exception ex) {
				DMLRuntimeException re = ex instanceof DMLRuntimeException ? (DMLRuntimeException) ex : new DMLRuntimeException(ex);

				for (OOCStream<?> q : queues) {
					q.propagateFailure(re);
				}

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
