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
import java.util.concurrent.atomic.AtomicIntegerArray;

import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCStoreLayout;
import org.apache.sysds.runtime.ooc.store.CountingLiveness;
import org.apache.sysds.runtime.ooc.store.IndexedMaterializedStoreReader;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public final class TSMMOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final MMTSJType _type;
	private final AggregateBinaryOperator _multiply;
	private final BinaryOperator _plus;
	private final AtomicBoolean _inputComplete = new AtomicBoolean();
	private final AtomicBoolean _liveInputEnded = new AtomicBoolean();
	private final AtomicBoolean _liveSchedulingReady = new AtomicBoolean();
	private final AtomicInteger _active = new AtomicInteger(1);
	private boolean _liveInput;
	private MaterializedStore<IndexedMatrixValue> _inputStore;
	private volatile IndexedMaterializedStoreReader<IndexedMatrixValue> _inputReader;
	private StateTable<IndexedMatrixValue> _accumulators;
	private AtomicIntegerArray _tilesSeen;
	private AtomicIntegerArray _tilesArrived;
	private AtomicIntegerArray _groupsScheduled;
	private OOCStream<AutoCloseable> _ready;
	private OOCStream<IndexedMatrixValue> _outputStream;
	private int _groups;
	private int _width;
	private long _taskBytes;

	public TSMMOOCPrimitive(OOCStreamable<IndexedMatrixValue> input, OOCStreamable<IndexedMatrixValue> output,
		MMTSJType type, AggregateBinaryOperator multiply, BinaryOperator plus, StreamContext context) {
		super(context, input);
		_output = output;
		_type = type;
		_multiply = multiply;
		_plus = plus;
	}

	@Override
	public List<OOCMaterializedInputRequest> requiredMaterializedInputs() {
		OOCStoreLayout layout = _type.isLeft() ? OOCStoreLayout.ROW_MAJOR : OOCStoreLayout.COL_MAJOR;
		return List.of(new OOCMaterializedInputRequest(0, layout, 1, this::accept, live -> _liveInput = live));
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = _type.isLeft() ? OOCAccessPattern.ROW_MAJOR : OOCAccessPattern.COL_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(_pattern);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = _type.isLeft() ? OOCAccessPattern.ROW_MAJOR : OOCAccessPattern.COL_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(_pattern);
	}

	private static long taskBytes(DataCharacteristics inputDc, DataCharacteristics outputDc) {
		long inputBytes = OOCUtils.estimateFullTileBytes(inputDc);
		long outputBytes = OOCUtils.estimateFullTileBytes(outputDc);
		long multiplyBytes = 2 * OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(inputBytes) + 2 * outputBytes;
		long mergeBytes = OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(outputBytes) + 2 * outputBytes;
		return Math.max(multiplyBytes, mergeBytes);
	}

	@Override
	protected void startExecution() {
		DataCharacteristics inputDc = getInput(0).getDataCharacteristics();
		if(inputDc == null || !inputDc.dimsKnown() || inputDc.getBlocksize() <= 0)
			throw new DMLRuntimeException("TSMM OOC requires known input dimensions and block size.");
		_groups = Math.toIntExact(_type.isLeft() ? inputDc.getNumRowBlocks() : inputDc.getNumColBlocks());
		_width = Math.toIntExact(_type.isLeft() ? inputDc.getNumColBlocks() : inputDc.getNumRowBlocks());
		if(_groups <= 0 || _width <= 0)
			throw new DMLRuntimeException("TSMM OOC requires non-empty input block geometry.");
		_tilesSeen = new AtomicIntegerArray(_groups);
		_tilesArrived = new AtomicIntegerArray(_groups * _width);
		_groupsScheduled = new AtomicIntegerArray(_groups);
		_accumulators = new StateTable<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID());
		long accumulatorPriorityOffset = (long) _width * _width;
		_accumulators.addEvictionPolicy(slot -> slot - accumulatorPriorityOffset);
		_outputStream = _output.getWriteStream();
		_ready = new SubscribableTaskQueue<>();

		_taskBytes = taskBytes(inputDc, _output.getDataCharacteristics());

		getContext().addOutStream(_outputStream, _ready);
		OOCInstructionUtils.submitCloseableOOCTasks(_ready, work -> {
			if(work instanceof MultiplyWork multiply)
				multiply(multiply);
			else
				merge((MergeWork) work);
		}, getContext()).whenComplete((ignored, error) -> {
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

		getMaterializedInput(0).whenComplete((store, error) -> {
			if(error != null) {
				fail(error);
				finishInput();
				return;
			}
			_inputStore = store;
			CountingLiveness liveness = new CountingLiveness(_groups * _width, _width + 1);
			if(_liveInput) {
				_inputReader = store.openLiveIndexedReader(liveness);
				for(int group = 0; group < _groups; group++)
					tryScheduleGroup(group);
				_liveSchedulingReady.set(true);
				finishLiveInput();
			}
			else {
				store.completion().whenComplete((ignored, completionError) -> {
					if(completionError != null) {
						fail(completionError);
						finishInput();
						return;
					}
					_inputReader = store.openIndexedReader(liveness);
					OOCInstructionUtils.submitOOCTask(this::drain, new StreamContext().addOutStream(_outputStream));
				});
			}
		});
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
			_liveInputEnded.set(true);
			finishLiveInput();
			return;
		}

		try(callback) {
			IndexedMatrixValue tile = callback.get();
			long rowIndex = tile.getIndexes().getRowIndex();
			long colIndex = tile.getIndexes().getColumnIndex();
			int group = Math.toIntExact((_type.isLeft() ? rowIndex : colIndex) - 1);
			int position = Math.toIntExact((_type.isLeft() ? colIndex : rowIndex) - 1);
			if(group < 0 || group >= _groups || position < 0 || position >= _width)
				throw new DMLRuntimeException("TSMM live tile " + tile.getIndexes() + " is outside the input geometry "
					+ _groups + "x" + _width + ".");
			if(!_tilesArrived.compareAndSet(group * _width + position, 0, 1))
				return;
			_tilesSeen.incrementAndGet(group);
			tryScheduleGroup(group);
		}
		catch(Throwable failure) {
			fail(failure);
			finishInput();
		}
	}

	private void finishLiveInput() {
		if(_liveInputEnded.get() && _liveSchedulingReady.get())
			finishInput();
	}

	private void drain() {
		try {
			for(int group = 0; group < _groups && !hasFailed(); group++)
				scheduleGroup(group);
		}
		catch(Throwable failure) {
			fail(failure);
		}
		finally {
			finishInput();
		}
	}

	private void tryScheduleGroup(int group) {
		if(_inputReader != null && _tilesSeen.get(group) == _width && _groupsScheduled.compareAndSet(group, 0, 1))
			scheduleGroup(group);
	}

	private void scheduleGroup(int group) {
		for(int left = 0; left < _width; left++)
			for(int right = left; right < _width; right++)
				schedulePair(group, left, right);
	}

	private void schedulePair(int group, int left, int right) {
		_active.incrementAndGet();
		_allowance.reserveAsync(_taskBytes).whenComplete((ignored, admissionError) -> {
			if(admissionError != null) {
				fail(admissionError);
				completeOne();
				return;
			}
			ReservationBudget budget = new ReservationBudget(_allowance, _taskBytes).enableReuse();
			long groupIndex = group + 1L;
			long leftIndex = left + 1L;
			long rightIndex = right + 1L;
			OOCFuture
				.allOf(List.of(
					_inputReader.request(_type.isLeft() ? groupIndex : leftIndex,
						_type.isLeft() ? leftIndex : groupIndex, budget),
					_inputReader.request(_type.isLeft() ? groupIndex : rightIndex,
						_type.isLeft() ? rightIndex : groupIndex, budget)),
					StoreLease::close)
				.whenComplete((inputs, inputError) -> {
					if(inputError != null) {
						budget.close();
						fail(inputError);
						completeOne();
						return;
					}
					try {
						if(inputs.get(0) == null || inputs.get(1) == null)
							throw new DMLRuntimeException("Missing buffered TSMM tiles for group " + (group + 1));
						_ready.enqueue(new MultiplyWork(group, left, right, inputs.get(0), inputs.get(1), budget));
					}
					catch(Throwable failure) {
						for(StoreLease<IndexedMatrixValue> lease : inputs)
							if(lease != null)
								lease.close();
						budget.close();
						fail(failure);
						completeOne();
					}
				});
		});
	}

	private void multiply(MultiplyWork work) {
		ReservationBudget budget = work.takeBudget();
		ManagedPayload<IndexedMatrixValue> partial = null;
		try {
			MatrixBlock left = (MatrixBlock) work._left.value().getValue();
			MatrixBlock right = (MatrixBlock) work._right.value().getValue();
			MatrixBlock block;
			if(work._leftPosition == work._rightPosition)
				block = left.transposeSelfMatrixMultOperations(new MatrixBlock(), _type);
			else if(_type.isLeft()) {
				MatrixBlock transposed = LibMatrixReorg.transpose(left);
				block = transposed.aggregateBinaryOperations(transposed, right, new MatrixBlock(), _multiply);
			}
			else {
				MatrixBlock transposed = LibMatrixReorg.transpose(right);
				block = left.aggregateBinaryOperations(left, transposed, new MatrixBlock(), _multiply);
			}
			int outputSlot = work._leftPosition * _width + work._rightPosition;
			partial = payload(outputSlot, 1, block, budget);
			OOCFuture<List<Void>> released = work.releaseInputsAsync();
			ManagedPayload<IndexedMatrixValue> result = partial;
			partial = null;
			released.whenComplete((ignored, error) -> {
				if(error != null) {
					result.release();
					budget.close();
					fail(error);
					completeOne();
				}
				else
					reduce(work._group, outputSlot, result, budget);
			});
		}
		catch(Throwable failure) {
			if(partial != null)
				partial.release();
			budget.close();
			fail(failure);
			completeOne();
		}
	}

	private void reduce(int group, int slot, ManagedPayload<IndexedMatrixValue> incoming, ReservationBudget budget) {
		if(count(incoming.value()) == _groups) {
			finalizeOutput(slot, incoming, budget);
			return;
		}
		OOCFuture<StoreLease<IndexedMatrixValue>> match;
		try {
			match = _accumulators.putOrTake(slot, incoming, budget);
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
				try {
					_ready.enqueue(new MergeWork(group, slot, incoming, existing, budget));
				}
				catch(Throwable failure) {
					incoming.release();
					existing.close();
					budget.close();
					fail(failure);
					completeOne();
				}
			}
		});
	}

	private void merge(MergeWork work) {
		ReservationBudget budget = work.takeBudget();
		ManagedPayload<IndexedMatrixValue> merged = null;
		try {
			IndexedMatrixValue existing = work._existing.value();
			IndexedMatrixValue incoming = work._incoming.value();
			MatrixBlock block = ((MatrixBlock) existing.getValue()).binaryOperations(_plus, incoming.getValue(),
				new MatrixBlock());
			merged = payload(work._slot, count(existing) + count(incoming), block, budget);
			work.releaseIncoming();
			OOCFuture<Void> released = work.closeExistingAsync();
			ManagedPayload<IndexedMatrixValue> result = merged;
			merged = null;
			released.whenComplete((ignored, error) -> {
				if(error != null) {
					result.release();
					budget.close();
					fail(error);
					completeOne();
				}
				else
					reduce(work._group, work._slot, result, budget);
			});
		}
		catch(Throwable failure) {
			if(merged != null)
				merged.release();
			budget.close();
			fail(failure);
			completeOne();
		}
	}

	private void finalizeOutput(int slot, ManagedPayload<IndexedMatrixValue> payload, ReservationBudget budget) {
		OOCStream.QueueCallback<IndexedMatrixValue> upper = null;
		OOCStream.QueueCallback<IndexedMatrixValue> lower = null;
		try {
			int row = slot / _width;
			int col = slot % _width;
			MatrixBlock block = (MatrixBlock) payload.value().getValue();
			payload.release();
			long upperBytes = block.getExactSerializedSize();
			budget.reserveBlocking(upperBytes);
			upper = new InMemoryQueueCallback<>(new IndexedMatrixValue(new MatrixIndexes(row + 1L, col + 1L), block),
				null, budget, upperBytes);
			if(row != col) {
				MatrixBlock mirror = LibMatrixReorg.transpose(block);
				long lowerBytes = mirror.getExactSerializedSize();
				budget.reserveBlocking(lowerBytes);
				lower = new InMemoryQueueCallback<>(
					new IndexedMatrixValue(new MatrixIndexes(col + 1L, row + 1L), mirror), null, budget, lowerBytes);
			}
			budget.close();
			_outputStream.enqueue(upper);
			upper = null;
			if(lower != null) {
				_outputStream.enqueue(lower);
				lower = null;
			}
		}
		catch(Throwable failure) {
			payload.release();
			budget.close();
			fail(failure);
		}
		finally {
			if(upper != null)
				upper.close();
			if(lower != null)
				lower.close();
		}
		completeOne();
	}

	private static ManagedPayload<IndexedMatrixValue> payload(int slot, int count, MatrixBlock block,
		ReservationBudget budget) {
		long bytes = block.getExactSerializedSize();
		budget.reserveBlocking(bytes);
		return new ManagedPayload<>(new IndexedMatrixValue(new MatrixIndexes(slot + 1L, count), block), bytes, budget);
	}

	private static int count(IndexedMatrixValue value) {
		return Math.toIntExact(value.getIndexes().getColumnIndex());
	}

	private void finishInput() {
		if(_inputComplete.compareAndSet(false, true))
			completeOne();
	}

	private void completeOne() {
		if(_active.decrementAndGet() != 0)
			return;
		try {
			_ready.closeInput();
		}
		catch(IllegalStateException ignored) {
		}
	}

	private void cleanup() {
		_accumulators.close();
		if(_inputReader != null)
			_inputReader.close();
		if(_inputStore != null)
			_inputStore.close();
		onComplete();
	}

	private static final class MultiplyWork implements AutoCloseable {
		private final int _group;
		private final int _leftPosition;
		private final int _rightPosition;
		private StoreLease<IndexedMatrixValue> _left;
		private StoreLease<IndexedMatrixValue> _right;
		private ReservationBudget _budget;

		private MultiplyWork(int group, int leftPosition, int rightPosition, StoreLease<IndexedMatrixValue> left,
			StoreLease<IndexedMatrixValue> right, ReservationBudget budget) {
			_group = group;
			_leftPosition = leftPosition;
			_rightPosition = rightPosition;
			_left = left;
			_right = right;
			_budget = budget;
		}

		private ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		private OOCFuture<List<Void>> releaseInputsAsync() {
			OOCFuture<Void> left = _left.closeAsync();
			OOCFuture<Void> right = _right.closeAsync();
			_left = null;
			_right = null;
			return OOCFuture.allOf(List.of(left, right), ignored -> {
			});
		}

		@Override
		public void close() {
			if(_left != null)
				_left.close();
			if(_right != null)
				_right.close();
			if(_budget != null)
				_budget.close();
		}
	}

	private static final class MergeWork implements AutoCloseable {
		private final int _group;
		private final int _slot;
		private ManagedPayload<IndexedMatrixValue> _incoming;
		private StoreLease<IndexedMatrixValue> _existing;
		private ReservationBudget _budget;

		private MergeWork(int group, int slot, ManagedPayload<IndexedMatrixValue> incoming,
			StoreLease<IndexedMatrixValue> existing, ReservationBudget budget) {
			_group = group;
			_slot = slot;
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
			_incoming.release();
			_incoming = null;
		}

		private OOCFuture<Void> closeExistingAsync() {
			OOCFuture<Void> released = _existing.closeAsync();
			_existing = null;
			return released;
		}

		@Override
		public void close() {
			if(_incoming != null)
				_incoming.release();
			if(_existing != null)
				_existing.close();
			if(_budget != null)
				_budget.close();
		}
	}
}
