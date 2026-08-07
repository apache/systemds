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

package org.apache.sysds.runtime.ooc.store;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.runtime.ooc.primitives.MaterializeOOCPrimitive;
import org.apache.sysds.runtime.ooc.primitives.OOCPrimitive;

public final class MaterializedStoreStreamable implements OOCStreamable<IndexedMatrixValue> {
	private static final int REPLAY_PREFETCH = 8;

	private final MaterializeOOCPrimitive _primitive;
	private MaterializedStore<IndexedMatrixValue> _store;
	private SyncMemoryAllowance _readerAllowance;
	private CacheableData<?> _data;
	private boolean _deleteScheduled;
	private boolean _materializationDone;
	private boolean _readersSealed;
	private boolean _closed;
	private int _reservedReaders;
	private int _pendingReaders;
	private int _activeReaders;

	public MaterializedStoreStreamable(OOCStream<IndexedMatrixValue> source, CacheableData<?> data) {
		if(source == null)
			throw new IllegalArgumentException("Materialized stream requires a source.");
		_data = data;
		_primitive = MaterializeOOCPrimitive.reusable(source);
		_primitive.store().whenComplete((store, error) -> {
			if(error != null) {
				markMaterializationDone();
				return;
			}
			synchronized(this) {
				_store = store;
			}
			store.completion().whenComplete((ignored, completionError) -> markMaterializationDone());
			tryFinalize();
		});
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReadStream() {
		return createReader(false);
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReservedReadStream() {
		return createReader(true);
	}

	private synchronized OOCStream<IndexedMatrixValue> createReader(boolean reserved) {
		if(reserved && _reservedReaders > 0)
			_reservedReaders--;
		else if(_deleteScheduled)
			throw new DMLRuntimeException("Cannot open a reader on a materialized stream scheduled for deletion.");
		_pendingReaders++;
		DeferredReader stream = new DeferredReader(this);
		stream.setData(_data);
		stream.assignPrimitive(_primitive);
		return stream;
	}

	private void openReader(DeferredReader output) {
		_primitive.store().whenComplete((store, storeError) -> {
			if(storeError != null) {
				failPendingReader(output, storeError);
				return;
			}
			store.completion().whenComplete((ignored, completionError) -> {
				if(completionError != null) {
					failPendingReader(output, completionError);
					return;
				}
				OrderedMaterializedStoreReader<IndexedMatrixValue> reader = null;
				try {
					reader = store.openReader(new SequentialAccessPattern(store.size()), readerAllowance(),
						REPLAY_PREFETCH);
					synchronized(this) {
						_pendingReaders--;
						_activeReaders++;
					}
					tryFinalize();
					drive(output, reader);
				}
				catch(Throwable failure) {
					if(reader == null)
						failPendingReader(output, failure);
					else {
						reader.close();
						try {
							output.propagateFailure(DMLRuntimeException.of(failure));
						}
						finally {
							finishReader(output);
						}
					}
				}
			});
		});
	}

	private void drive(DeferredReader output, OrderedMaterializedStoreReader<IndexedMatrixValue> reader) {
		StoreBackedStream<IndexedMatrixValue> replay = new StoreBackedStream<>(reader);
		replay.setData(_data);
		replay.setSubscriber(callback -> {
			if(callback.isFailure()) {
				DMLRuntimeException failure;
				try {
					callback.get();
					failure = new DMLRuntimeException("Materialized replay failed.");
				}
				catch(Throwable error) {
					failure = DMLRuntimeException.of(error);
				}
				try {
					output.propagateFailure(failure);
				}
				finally {
					finishReader(output);
				}
			}
			else if(callback.isEos()) {
				try {
					output.closeInput();
				}
				finally {
					finishReader(output);
				}
			}
			else {
				OOCStream.QueueCallback<IndexedMatrixValue> retained = callback.keepOpen();
				try {
					output.enqueue(retained);
				}
				catch(Throwable failure) {
					retained.close();
					throw DMLRuntimeException.of(failure);
				}
			}
		});
	}

	private void failPendingReader(DeferredReader output, Throwable error) {
		if(!output.finish())
			return;
		synchronized(this) {
			_pendingReaders--;
		}
		try {
			output.propagateFailure(DMLRuntimeException.of(error));
		}
		finally {
			tryFinalize();
		}
	}

	private void finishReader(DeferredReader output) {
		if(!output.finish())
			return;
		synchronized(this) {
			_activeReaders--;
		}
		tryFinalize();
	}

	private synchronized SyncMemoryAllowance readerAllowance() {
		if(_readerAllowance == null)
			_readerAllowance = new SyncMemoryAllowance(GlobalMemoryBroker.get());
		return _readerAllowance;
	}

	private void markMaterializationDone() {
		synchronized(this) {
			_materializationDone = true;
		}
		tryFinalize();
	}

	@Override
	public synchronized void reserveLazyHandle() {
		if(_closed || (_deleteScheduled && _reservedReaders == 0))
			throw new DMLRuntimeException("Cannot reserve a reader on a closed materialized stream.");
		_reservedReaders++;
	}

	@Override
	public void discardHandle() {
		synchronized(this) {
			if(_reservedReaders <= 0)
				return;
			_reservedReaders--;
		}
		tryFinalize();
	}

	@Override
	public void scheduleMaterializedStoreDeletion() {
		synchronized(this) {
			_deleteScheduled = true;
		}
		tryFinalize();
	}

	private void tryFinalize() {
		MaterializedStore<IndexedMatrixValue> store;
		SyncMemoryAllowance allowance = null;
		boolean seal = false;
		boolean close = false;
		synchronized(this) {
			store = _store;
			if(!_deleteScheduled || _reservedReaders != 0 || _pendingReaders != 0)
				return;
			if(store != null && !_readersSealed) {
				_readersSealed = true;
				seal = true;
			}
			if(_materializationDone && _activeReaders == 0 && !_closed) {
				_closed = true;
				close = store != null;
				allowance = _readerAllowance;
			}
		}
		if(seal)
			store.sealReaders();
		if(close)
			store.close();
		if(allowance != null)
			allowance.shutdown();
	}

	@Override
	public boolean hasMaterializedStore() {
		return true;
	}

	@Override
	public OOCStream<IndexedMatrixValue> getWriteStream() {
		throw new UnsupportedOperationException("Materialized streams are read-only.");
	}

	@Override
	public boolean hasStreamCache() {
		return false;
	}

	@Override
	public CachingStream getStreamCache() {
		return null;
	}

	@Override
	public boolean isProcessed() {
		return false;
	}

	@Override
	public synchronized DataCharacteristics getDataCharacteristics() {
		return _data == null ? null : _data.getDataCharacteristics();
	}

	@Override
	public synchronized CacheableData<?> getData() {
		return _data;
	}

	@Override
	public synchronized void setData(CacheableData<?> data) {
		_data = data;
	}

	@Override
	public OOCPrimitive getPrimitive() {
		return _primitive;
	}

	private static final class DeferredReader extends SubscribableTaskQueue<IndexedMatrixValue> {
		private final MaterializedStoreStreamable _owner;
		private final AtomicBoolean _activated;
		private final AtomicBoolean _finished;

		private DeferredReader(MaterializedStoreStreamable owner) {
			_owner = owner;
			_activated = new AtomicBoolean();
			_finished = new AtomicBoolean();
		}

		@Override
		public void setSubscriber(Consumer<QueueCallback<IndexedMatrixValue>> subscriber) {
			super.setSubscriber(subscriber);
			activate();
		}

		@Override
		public IndexedMatrixValue dequeue() {
			activate();
			return super.dequeue();
		}

		@Override
		public QueueCallback<IndexedMatrixValue> dequeueCB() {
			activate();
			return super.dequeueCB();
		}

		private void activate() {
			if(_activated.compareAndSet(false, true))
				_owner.openReader(this);
		}

		private boolean finish() {
			return _finished.compareAndSet(false, true);
		}
	}
}
