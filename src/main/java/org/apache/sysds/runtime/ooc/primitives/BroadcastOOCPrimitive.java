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

package org.apache.sysds.runtime.ooc.primitives;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import java.util.function.ToIntFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCStoreLayout;
import org.apache.sysds.runtime.ooc.store.IndexedMaterializedStoreReader;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.AllocatedOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public final class BroadcastOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _broadcast;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final ToIntFunction<IndexedMatrixValue> _lookup;
	private final Supplier<MaterializedStore.Liveness> _liveness;
	private final BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> _operation;
	private final AtomicBoolean _cleaned;
	private final AtomicBoolean _sourceComplete;
	private final AtomicInteger _active;
	private MaterializedStore<IndexedMatrixValue> _store;
	private IndexedMaterializedStoreReader<IndexedMatrixValue> _reader;
	private OOCStream<BroadcastWork> _ready;
	private OOCStream<IndexedMatrixValue> _outputStream;

	public BroadcastOOCPrimitive(OOCStreamable<IndexedMatrixValue> streamed,
		OOCStreamable<IndexedMatrixValue> broadcast, OOCStreamable<IndexedMatrixValue> output,
		ToIntFunction<IndexedMatrixValue> lookup, Supplier<MaterializedStore.Liveness> liveness,
		BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> operation, StreamContext context) {
		super(context, streamed, broadcast);
		_broadcast = broadcast;
		_output = output;
		_lookup = lookup;
		_liveness = liveness;
		_operation = operation;
		_cleaned = new AtomicBoolean();
		_sourceComplete = new AtomicBoolean();
		_active = new AtomicInteger(1);
	}

	@Override
	public List<OOCMaterializedInputRequest> requiredMaterializedInputs() {
		return List.of(new OOCMaterializedInputRequest(1, OOCStoreLayout.ROW_MAJOR, 1));
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ROW_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(OOCAccessPattern.ROW_MAJOR);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = OOCAccessPattern.ROW_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(OOCAccessPattern.ROW_MAJOR);
	}

	@Override
	protected void startExecution() {
		_outputStream = _output.getWriteStream();
		_ready = new SubscribableTaskQueue<>();
		getContext().addOutStream(_outputStream, _ready);
		OOCInstructionUtils.submitCloseableOOCTasks(_ready, this::process, getContext())
			.whenComplete((ignored, error) -> {
				try {
					if(error != null)
						fail(error);
					_outputStream.closeInput();
				}
				catch(Throwable failure) {
					fail(failure);
				}
				finally {
					cleanup();
				}
			});

		getMaterializedInput(1).whenComplete((store, error) -> {
			if(error != null) {
				fail(error);
				finishSource();
				return;
			}
			_store = store;
			store.completion().whenComplete((ignored, completionError) -> {
				if(completionError != null) {
					fail(completionError);
					finishSource();
					return;
				}
				try {
					_reader = store.openIndexedReader(_liveness.get());
					startBroadcast();
				}
				catch(Throwable failure) {
					fail(failure);
					finishSource();
				}
			});
		});
	}

	private void startBroadcast() {
		long broadcastLogical = OOCUtils.estimateFullTileBytes(_broadcast.getDataCharacteristics());
		long outputLogical = OOCUtils.estimateFullTileBytes(_output.getDataCharacteristics());
		long broadcastPin = OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(broadcastLogical);
		long taskBytes = broadcastPin * 2 + outputLogical * 2;
		OOCStream<IndexedMatrixValue> streamed = getInputReadStream(0);
		AllocatedOOCStream<IndexedMatrixValue> admitted = new AllocatedOOCStream<>(streamed, _allowance,
			ignored -> taskBytes);
		getContext().addInStream(streamed, admitted);
		admitted.setSubscriber(this::accept);
	}

	private void accept(OOCStream.QueueCallback<IndexedMatrixValue> callback) {
		if(callback.isEos() || callback.isFailure()) {
			try(callback) {
				if(callback.isFailure())
					callback.get();
			}
			catch(Throwable failure) {
				fail(failure);
			}
			finishSource();
			return;
		}

		ReservationBudget budget = null;
		OOCStream.QueueCallback<IndexedMatrixValue> retained = null;
		_active.incrementAndGet();
		try(callback) {
			budget = AllocatedOOCStream.detachBudget(callback);
			if(budget == null)
				throw new DMLRuntimeException("Missing admitted broadcast task budget.");
			IndexedMatrixValue streamed = callback.get();
			int lookup = _lookup.applyAsInt(streamed);
			retained = callback.keepOpen();
			OOCFuture<StoreLease<IndexedMatrixValue>> requested = _reader.request(lookup, budget);
			OOCStream.QueueCallback<IndexedMatrixValue> pendingStreamed = retained;
			ReservationBudget pendingBudget = budget;
			retained = null;
			budget = null;
			requested.whenComplete(
				(broadcast, error) -> broadcastReady(pendingStreamed, broadcast, pendingBudget, lookup, error));
		}
		catch(Throwable failure) {
			fail(failure);
			completeOne();
		}
		finally {
			if(retained != null)
				retained.close();
			if(budget != null)
				budget.close();
		}
	}

	private void broadcastReady(OOCStream.QueueCallback<IndexedMatrixValue> streamed,
		StoreLease<IndexedMatrixValue> broadcast, ReservationBudget budget, int lookup, Throwable error) {
		if(error != null || broadcast == null) {
			try {
				streamed.close();
				if(broadcast != null)
					broadcast.close();
				budget.close();
			}
			finally {
				fail(error != null ? error : new IllegalStateException("Missing broadcast tile " + lookup));
				completeOne();
			}
			return;
		}
		BroadcastWork work = new BroadcastWork(streamed, broadcast, budget);
		try {
			_ready.enqueue(work);
		}
		catch(Throwable failure) {
			work.close();
			fail(failure);
			completeOne();
		}
	}

	private void process(BroadcastWork work) {
		ReservationBudget budget = work.takeBudget();
		try {
			IndexedMatrixValue output = _operation.apply(work._streamed.get(), work._broadcast.value());
			OOCUtils.enqueueExact(_outputStream, output, budget);
			budget = null;
		}
		catch(Throwable failure) {
			fail(failure);
		}
		finally {
			if(budget != null)
				budget.close();
			completeOne();
		}
	}

	private void finishSource() {
		if(_sourceComplete.compareAndSet(false, true))
			completeOne();
	}

	private void completeOne() {
		if(_active.decrementAndGet() != 0)
			return;
		try {
			_ready.closeInput();
		}
		catch(IllegalStateException ignored) {
			// Failure propagation may already have closed the ready stream.
		}
	}

	private void cleanup() {
		if(!_cleaned.compareAndSet(false, true))
			return;
		try {
			if(_reader != null)
				_reader.close();
		}
		finally {
			try {
				if(_store != null)
					_store.close();
			}
			finally {
				onComplete();
			}
		}
	}

	private static final class BroadcastWork implements AutoCloseable {
		private OOCStream.QueueCallback<IndexedMatrixValue> _streamed;
		private StoreLease<IndexedMatrixValue> _broadcast;
		private ReservationBudget _budget;

		private BroadcastWork(OOCStream.QueueCallback<IndexedMatrixValue> streamed,
			StoreLease<IndexedMatrixValue> broadcast, ReservationBudget budget) {
			_streamed = streamed;
			_broadcast = broadcast;
			_budget = budget;
		}

		private ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		@Override
		public void close() {
			if(_streamed != null) {
				_streamed.close();
				_streamed = null;
			}
			if(_broadcast != null) {
				_broadcast.close();
				_broadcast = null;
			}
			if(_budget != null) {
				_budget.close();
				_budget = null;
			}
		}
	}
}
