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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;
import org.apache.sysds.runtime.ooc.util.StateTableUtils;

public class JoinOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _left;
	private final OOCStreamable<IndexedMatrixValue> _right;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final BiFunction<MatrixBlock, MatrixBlock, MatrixBlock> _operation;
	private final AtomicInteger _pending = new AtomicInteger(1);
	private final AtomicInteger _unmatched = new AtomicInteger();
	private final CompletableFuture<Void> _pendingCompletion = new CompletableFuture<>();
	private StateTable<IndexedMatrixValue> _table;

	public JoinOOCPrimitive(OOCStreamable<IndexedMatrixValue> left, OOCStreamable<IndexedMatrixValue> right,
		OOCStreamable<IndexedMatrixValue> output, BiFunction<MatrixBlock, MatrixBlock, MatrixBlock> operation,
		StreamContext context) {
		super(context, left, right);
		_left = left;
		_right = right;
		_output = output;
		_operation = operation;
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ANY;
		for(OOCPrimitive child : getChildren())
			_pattern = _pattern.fused(child.getAccessPattern());
		if(_pattern.isPlannable() && _pattern != OOCAccessPattern.ANY)
			for(OOCPrimitive child : getChildren())
				child.requestPattern(_pattern);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = accessPattern;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(accessPattern);
	}

	@Override
	protected void startExecution() {
		OOCStream<IndexedMatrixValue> left = getInputReadStream(0);
		OOCStream<IndexedMatrixValue> right = getInputReadStream(1);
		_table = new StateTable<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID());
		OOCStream<IndexedMatrixValue> output = _output.getWriteStream();
		OOCStream<JoinWork> matches = new SubscribableTaskQueue<>();
		long inputBytes = Math.max(OOCUtils.estimateOutputTileBytes(_left.getDataCharacteristics()),
			OOCUtils.estimateOutputTileBytes(_right.getDataCharacteristics()));
		long outputBytes = OOCUtils.estimateOutputTileBytes(_output.getDataCharacteristics());
		long taskBytes = outputBytes + 2 * inputBytes;

		getContext().addOutStream(output);
		CompletableFuture<Void> processing = OOCInstructionUtils.submitCloseableOOCTasks(matches, (JoinWork work) -> {
			IndexedMatrixValue mleft = work._left.get();
			IndexedMatrixValue mright = work._right.get();
			OOCUtils.enqueueExact(output, new IndexedMatrixValue(mleft.getIndexes(),
				_operation.apply((MatrixBlock) mleft.getValue(), (MatrixBlock) mright.getValue())), work._budget);
		}, getContext());
		CompletableFuture.allOf(processing, _pendingCompletion).thenRun(() -> {
			try {
				_table.close();
				onComplete();
			}
			finally {
				output.closeInput();
			}
		});

		OOCInstructionUtils.submitOOCTask(() -> drive(left, right, matches, taskBytes),
			new StreamContext().addOutStream(output));
	}

	private void drive(OOCStream<IndexedMatrixValue> leftInput, OOCStream<IndexedMatrixValue> rightInput,
		OOCStream<JoinWork> matches, long taskBytes) {
		long cols = _right.getDataCharacteristics().getNumColBlocks();
		try {
			while(true) {
				OOCStream.QueueCallback<IndexedMatrixValue> left = leftInput.dequeueCB();
				OOCStream.QueueCallback<IndexedMatrixValue> right = rightInput.dequeueCB();
				boolean leftEos = left == null || left.isEos();
				boolean rightEos = right == null || right.isEos();
				if(leftEos || rightEos) {
					if(left != null)
						left.close();
					if(right != null)
						right.close();
					if(leftEos != rightEos)
						throw new DMLRuntimeException("Join inputs contain a different number of blocks");
					break;
				}
				accept(left, true, cols, taskBytes, matches);
				accept(right, false, cols, taskBytes, matches);
			}
		}
		catch(Throwable failure) {
			fail(failure);
			throw DMLRuntimeException.of(failure);
		}
		finally {
			completePending(matches);
		}
	}

	private void accept(OOCStream.QueueCallback<IndexedMatrixValue> callback, boolean left, long cols, long taskBytes,
		OOCStream<JoinWork> matches) {
		if(callback == null)
			return;
		OOCStream.QueueCallback<IndexedMatrixValue> owned = null;
		ReservationBudget budget = null;
		boolean pending = false;
		try {
			owned = callback.keepOpen();
			callback.close();
			callback = null;
			budget = OOCUtils.reserveBudget(_allowance, taskBytes);
			IndexedMatrixValue value = owned.get();
			long row = value.getIndexes().getRowIndex() - 1;
			long col = value.getIndexes().getColumnIndex() - 1;
			int slot = Math.toIntExact(row * cols + col);
			_pending.incrementAndGet();
			pending = true;
			OOCFuture<StateTableUtils.Match> future = StateTableUtils.putOrTake(_table, slot, owned, budget);
			owned = null;
			ReservationBudget pendingBudget = budget;
			budget = null;
			future.whenComplete((match, error) -> matchReady(match, left, pendingBudget, error, matches));
			pending = false;
		}
		finally {
			if(pending)
				completePending(matches);
			if(callback != null)
				callback.close();
			if(owned != null)
				owned.close();
			if(budget != null)
				budget.close();
		}
	}

	private void matchReady(StateTableUtils.Match match, boolean left, ReservationBudget budget, Throwable error,
		OOCStream<JoinWork> matches) {
		JoinWork work = null;
		try {
			if(error != null)
				throw DMLRuntimeException.of(error);
			if(match == null) {
				_unmatched.incrementAndGet();
				return;
			}
			_unmatched.decrementAndGet();
			work = left ? new JoinWork(match.left(), match.right(), budget) : new JoinWork(match.right(), match.left(),
				budget);
			match = null;
			budget = null;
			matches.enqueue(work);
			work = null;
		}
		catch(Throwable failure) {
			fail(failure);
		}
		finally {
			if(work != null)
				work.close();
			if(match != null) {
				match.left().close();
				match.right().close();
			}
			if(budget != null)
				budget.close();
			completePending(matches);
		}
	}

	private void completePending(OOCStream<JoinWork> matches) {
		if(_pending.decrementAndGet() != 0)
			return;
		try {
			int unmatched = _unmatched.get();
			if(unmatched != 0)
				fail(new DMLRuntimeException("Join inputs contain " + unmatched + " unmatched blocks"));
			else {
				try {
					matches.closeInput();
				}
				catch(Exception ignored) {
				}
			}
		}
		finally {
			_pendingCompletion.complete(null);
		}
	}

	private static final class JoinWork implements AutoCloseable {
		private final OOCStream.QueueCallback<IndexedMatrixValue> _left;
		private final OOCStream.QueueCallback<IndexedMatrixValue> _right;
		private final ReservationBudget _budget;

		private JoinWork(OOCStream.QueueCallback<IndexedMatrixValue> left,
			OOCStream.QueueCallback<IndexedMatrixValue> right, ReservationBudget budget) {
			_left = left;
			_right = right;
			_budget = budget;
		}

		@Override
		public void close() {
			try(_left; _right; _budget) {
				// Release
			}
		}
	}
}
