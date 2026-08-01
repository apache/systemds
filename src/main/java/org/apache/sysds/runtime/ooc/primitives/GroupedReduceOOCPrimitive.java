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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.AllocatedOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public final class GroupedReduceOOCPrimitive extends OOCPrimitive {
	private final OOCStream<IndexedMatrixValue> _input;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final BiFunction<MatrixBlock, MatrixBlock, MatrixBlock> _merge;
	private final AtomicBoolean _cleaned;
	private final AtomicBoolean _failed;
	private final AtomicBoolean _sourceComplete;
	private final AtomicInteger _active;
	private final AtomicInteger _finalizedGroups;
	private StateTable<IndexedMatrixValue> _table;
	private OOCStream<MergeWork> _ready;
	private OOCStream<IndexedMatrixValue> _outputStream;
	private int _numGroups;
	private int _groupSize;

	public GroupedReduceOOCPrimitive(OOCStreamable<IndexedMatrixValue> input, OOCStreamable<IndexedMatrixValue> output,
		BiFunction<MatrixBlock, MatrixBlock, MatrixBlock> merge, StreamContext context) {
		this(input.getReadStream(), output, merge, context);
	}

	private GroupedReduceOOCPrimitive(OOCStream<IndexedMatrixValue> input, OOCStreamable<IndexedMatrixValue> output,
		BiFunction<MatrixBlock, MatrixBlock, MatrixBlock> merge, StreamContext context) {
		super(context, input.getPrimitive() == null ? List.of() : List.of(input.getPrimitive()));
		_input = input;
		_output = output;
		_merge = merge;
		_cleaned = new AtomicBoolean();
		_failed = new AtomicBoolean();
		_sourceComplete = new AtomicBoolean();
		_active = new AtomicInteger(1);
		_finalizedGroups = new AtomicInteger();
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
		DataCharacteristics inputDc = _input.getDataCharacteristics();
		if(inputDc == null || !inputDc.dimsKnown() || inputDc.getBlocksize() <= 0)
			throw new DMLRuntimeException("Grouped OOC reduction requires known input dimensions and block size.");
		_numGroups = Math.toIntExact(inputDc.getNumRowBlocks());
		_groupSize = Math.toIntExact(inputDc.getNumColBlocks());
		_outputStream = _output.getWriteStream();
		_ready = new SubscribableTaskQueue<>();
		getContext().addInStream(_input).addOutStream(_outputStream, _ready);
		_table = new StateTable<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID());

		OOCInstructionUtils.submitOOCTasks(_ready, callback -> process(callback.get()), getContext())
			.whenComplete((ignored, error) -> {
				try {
					_outputStream.closeInput();
				}
				catch(Throwable failure) {
					fail(failure);
				}
				finally {
					cleanup();
				}
			});

		long logicalBytes = Math.max(OOCUtils.estimateFullTileBytes(inputDc),
			OOCUtils.estimateFullTileBytes(_output.getDataCharacteristics()));
		long pinBytes = OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(logicalBytes);
		long taskBytes = pinBytes + logicalBytes * 2;
		AllocatedOOCStream<IndexedMatrixValue> admitted = new AllocatedOOCStream<>(_input, _allowance,
			ignored -> taskBytes);
		getContext().addInStream(admitted);
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
		ManagedPayload<IndexedMatrixValue> payload = null;
		_active.incrementAndGet();
		try(callback) {
			budget = AllocatedOOCStream.detachBudget(callback).enableReuse();
			IndexedMatrixValue input = callback.get();
			int group = Math.toIntExact(input.getIndexes().getRowIndex() - 1);
			if(group < 0 || group >= _numGroups)
				throw new DMLRuntimeException("Invalid grouped-reduce row block: " + (group + 1));
			IndexedMatrixValue value = new IndexedMatrixValue(new MatrixIndexes(group + 1L, 1), input.getValue());
			payload = payload(value, budget);
			reduce(group, payload, budget);
			payload = null;
			budget = null;
		}
		catch(Throwable failure) {
			fail(failure);
			completeOne();
		}
		finally {
			if(payload != null)
				payload.release();
			if(budget != null)
				budget.close();
		}
	}

	private void reduce(int group, ManagedPayload<IndexedMatrixValue> incoming, ReservationBudget budget) {
		if(multiplicity(incoming.value()) == _groupSize) {
			finalizeGroup(group, incoming, budget);
			return;
		}
		OOCFuture<StoreLease<IndexedMatrixValue>> match;
		try {
			match = _table.putOrTake(group, incoming, budget);
		}
		catch(Throwable failure) {
			incoming.release();
			budget.close();
			fail(failure);
			completeOne();
			return;
		}
		match.whenComplete((existing, error) -> {
			if(error != null) {
				incoming.release();
				budget.close();
				fail(error);
				completeOne();
			}
			else if(existing == null) {
				budget.close();
				completeOne();
			}
			else {
				MergeWork work = new MergeWork(group, incoming, existing, budget);
				try {
					_ready.enqueue(work);
				}
				catch(Throwable failure) {
					work.close();
					fail(failure);
					completeOne();
				}
			}
		});
	}

	private void process(MergeWork work) {
		ReservationBudget budget = work.takeBudget();
		ManagedPayload<IndexedMatrixValue> merged = null;
		OOCFuture<Void> released;
		try {
			IndexedMatrixValue left = work._existing.value();
			IndexedMatrixValue right = work._incoming.value();
			int count = Math.addExact(multiplicity(left), multiplicity(right));
			if(count > _groupSize)
				throw new DMLRuntimeException("Too many partial tiles for grouped-reduce row " + (work._group + 1));
			MatrixBlock value = _merge.apply((MatrixBlock) left.getValue(), (MatrixBlock) right.getValue());
			merged = payload(new IndexedMatrixValue(new MatrixIndexes(work._group + 1L, count), value), budget);
			work.releaseIncoming();
			released = work.closeExistingAsync();
		}
		catch(Throwable failure) {
			if(merged != null)
				merged.release();
			work.close();
			budget.close();
			fail(failure);
			completeOne();
			return;
		}

		ManagedPayload<IndexedMatrixValue> next = merged;
		released.whenComplete((ignored, error) -> {
			if(error != null) {
				next.release();
				budget.close();
				fail(error);
				completeOne();
			}
			else
				reduce(work._group, next, budget);
		});
	}

	private void finalizeGroup(int group, ManagedPayload<IndexedMatrixValue> payload, ReservationBudget budget) {
		IndexedMatrixValue accumulated = payload.value();
		IndexedMatrixValue output = new IndexedMatrixValue(new MatrixIndexes(group + 1L, 1), accumulated.getValue());
		payload.release();
		try {
			OOCUtils.enqueueExact(_outputStream, output, budget);
			_finalizedGroups.incrementAndGet();
		}
		catch(Throwable failure) {
			budget.close();
			fail(failure);
		}
		completeOne();
	}

	private static ManagedPayload<IndexedMatrixValue> payload(IndexedMatrixValue value, ReservationBudget budget) {
		long bytes = ((MatrixBlock) value.getValue()).getExactSerializedSize();
		budget.reserveBlocking(bytes);
		return new ManagedPayload<>(value, bytes, budget);
	}

	private static int multiplicity(IndexedMatrixValue value) {
		return Math.toIntExact(value.getIndexes().getColumnIndex());
	}

	private void finishSource() {
		if(_sourceComplete.compareAndSet(false, true))
			completeOne();
	}

	private void completeOne() {
		int remaining = _active.decrementAndGet();
		if(remaining != 0)
			return;
		if(!_failed.get() && _finalizedGroups.get() != _numGroups)
			fail(new DMLRuntimeException(
				"Grouped reduction completed " + _finalizedGroups.get() + " of " + _numGroups + " row groups."));
		try {
			_ready.closeInput();
		}
		catch(IllegalStateException ignored) {
			// Failure propagation may already have closed the ready stream.
		}
	}

	private void fail(Throwable error) {
		if(!_failed.compareAndSet(false, true))
			return;
		DMLRuntimeException failure = DMLRuntimeException.of(error);
		_outputStream.propagateFailure(failure);
		getContext().failAll(failure);
	}

	private void cleanup() {
		if(!_cleaned.compareAndSet(false, true))
			return;
		try {
			if(_table != null)
				_table.close();
		}
		finally {
			onComplete();
		}
	}

	private static final class MergeWork implements AutoCloseable {
		private final int _group;
		private ManagedPayload<IndexedMatrixValue> _incoming;
		private StoreLease<IndexedMatrixValue> _existing;
		private ReservationBudget _budget;

		private MergeWork(int group, ManagedPayload<IndexedMatrixValue> incoming,
			StoreLease<IndexedMatrixValue> existing, ReservationBudget budget) {
			_group = group;
			_incoming = incoming;
			_existing = existing;
			_budget = budget;
		}

		private ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		private void releaseIncoming() {
			if(_incoming != null) {
				_incoming.release();
				_incoming = null;
			}
		}

		private OOCFuture<Void> closeExistingAsync() {
			StoreLease<IndexedMatrixValue> existing = _existing;
			_existing = null;
			return existing.closeAsync();
		}

		@Override
		public void close() {
			releaseIncoming();
			if(_existing != null) {
				_existing.close();
				_existing = null;
			}
			if(_budget != null)
				_budget.close();
		}
	}
}
