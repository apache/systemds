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

package org.apache.sysds.runtime.ooc.stream;

import java.util.function.ToLongFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;

public final class AllocatedOOCStream<T> extends SubscribableTaskQueue<T> {
	private final OOCStream<T> _source;
	private final MemoryAllowance _allowance;
	private final ToLongFunction<T> _reservationSize;
	private volatile DMLRuntimeException _failure;
	private int _pendingReservations;
	private boolean _sourceComplete;
	private boolean _outputClosed;

	public AllocatedOOCStream(OOCStream<T> source, MemoryAllowance allowance, ToLongFunction<T> reservationSize) {
		_source = source;
		_allowance = allowance;
		_reservationSize = reservationSize;
		setData(source.getData());
		source.setSubscriber(this::admit);
	}

	public static ReservationBudget detachBudget(OOCStream.QueueCallback<?> callback) {
		return callback instanceof BudgetedQueueCallback<?> budgeted ? budgeted.detachBudget() : null;
	}

	private void admit(OOCStream.QueueCallback<T> callback) {
		if(callback.isFailure()) {
			try(callback) {
				callback.get();
			}
			catch(DMLRuntimeException failure) {
				fail(failure);
			}
			finishSource();
			return;
		}
		if(callback.isEos()) {
			callback.close();
			finishSource();
			return;
		}
		if(_failure != null) {
			callback.close();
			return;
		}
		try(callback) {
			long bytes = _reservationSize.applyAsLong(callback.get());
			if(bytes < 0)
				throw new IllegalArgumentException("Cannot reserve negative bytes: " + bytes);
			if(bytes == 0) {
				enqueueOwned(callback.keepOpen(), null);
				return;
			}
			if(_allowance.tryReserve(bytes)) {
				enqueueOwned(callback.keepOpen(), new ReservationBudget(_allowance, bytes));
				return;
			}
			retainUntilAllocated(callback, bytes);
		}
		catch(RuntimeException error) {
			fail(DMLRuntimeException.of(error));
		}
	}

	private void retainUntilAllocated(OOCStream.QueueCallback<T> callback, long bytes) {
		OOCStream.QueueCallback<T> retained = callback.keepOpen();
		OOCFuture<Void> reservation;
		synchronized(this) {
			_pendingReservations++;
		}
		try {
			reservation = _allowance.reserveAsync(bytes);
		}
		catch(RuntimeException error) {
			try {
				retained.close();
			}
			finally {
				releasePendingReservation();
			}
			throw error;
		}
		reservation.whenComplete((ignored, error) -> {
			try {
				if(error != null) {
					fail(DMLRuntimeException.of(error));
					retained.close();
				}
				else if(_failure != null) {
					_allowance.release(bytes);
					retained.close();
				}
				else
					enqueueOwned(retained, new ReservationBudget(_allowance, bytes));
			}
			catch(RuntimeException completionError) {
				fail(DMLRuntimeException.of(completionError));
			}
			finally {
				releasePendingReservation();
			}
		});
	}

	private void enqueueOwned(OOCStream.QueueCallback<T> callback, ReservationBudget budget) {
		OOCStream.QueueCallback<T> output = budget == null ? callback : new BudgetedQueueCallback<>(callback, budget);
		try {
			enqueue(output);
		}
		catch(RuntimeException error) {
			output.close();
			throw error;
		}
	}

	private boolean fail(DMLRuntimeException failure) {
		synchronized(this) {
			if(_failure != null)
				return false;
			_failure = failure;
		}
		super.propagateFailure(failure);
		return true;
	}

	private void releasePendingReservation() {
		boolean close;
		synchronized(this) {
			if(_pendingReservations <= 0)
				throw new IllegalStateException("Pending reservation count underflow");
			_pendingReservations--;
			close = _sourceComplete && _pendingReservations == 0 && !_outputClosed;
			if(close)
				_outputClosed = true;
		}
		if(close)
			closeInput();
	}

	private void finishSource() {
		boolean close;
		synchronized(this) {
			if(_sourceComplete)
				return;
			_sourceComplete = true;
			close = _pendingReservations == 0 && !_outputClosed;
			if(close)
				_outputClosed = true;
		}
		if(close)
			closeInput();
	}

	@Override
	public void propagateFailure(DMLRuntimeException failure) {
		if(fail(failure))
			_source.propagateFailure(failure);
	}

	private static final class BudgetedQueueCallback<T> implements OOCStream.QueueCallback<T> {
		private final OOCStream.QueueCallback<T> _callback;
		private final BudgetedQueueCallback<T> _budgetOwner;
		private ReservationBudget _budget;
		private int _budgetReferences;
		private boolean _closed;

		private BudgetedQueueCallback(OOCStream.QueueCallback<T> callback, ReservationBudget budget) {
			_callback = callback;
			_budgetOwner = this;
			_budget = budget;
			_budgetReferences = 1;
		}

		private BudgetedQueueCallback(OOCStream.QueueCallback<T> callback, BudgetedQueueCallback<T> budgetOwner) {
			_callback = callback;
			_budgetOwner = budgetOwner;
		}

		private synchronized ReservationBudget detachBudget() {
			if(_closed)
				throw new IllegalStateException("Cannot detach from a closed callback");
			return _budgetOwner.takeBudget();
		}

		private synchronized ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		@Override
		public T get() {
			return _callback.get();
		}

		@Override
		public synchronized OOCStream.QueueCallback<T> keepOpen() {
			if(_closed)
				throw new IllegalStateException("Cannot keep open a closed callback");
			OOCStream.QueueCallback<T> retained = _callback.keepOpen();
			_budgetOwner.retainBudget();
			return new BudgetedQueueCallback<>(retained, _budgetOwner);
		}

		private synchronized void retainBudget() {
			_budgetReferences++;
		}

		@Override
		public void close() {
			synchronized(this) {
				if(_closed)
					return;
				_closed = true;
			}
			try {
				_callback.close();
			}
			finally {
				_budgetOwner.releaseBudget();
			}
		}

		private void releaseBudget() {
			ReservationBudget budget = null;
			synchronized(this) {
				_budgetReferences--;
				if(_budgetReferences == 0) {
					budget = _budget;
					_budget = null;
				}
			}
			if(budget != null)
				budget.close();
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			_callback.fail(failure);
		}

		@Override
		public boolean isEos() {
			return _callback.isEos();
		}

		@Override
		public boolean isFailure() {
			return _callback.isFailure();
		}
	}
}
