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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.ToIntFunction;

public final class OOCStreamMaterializer implements Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> {
	private final MaterializedStore<IndexedMatrixValue> _store;
	private final ToIntFunction<MatrixIndexes> _linearize;
	private final MemoryAllowance _allowance;
	private final List<Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>> _liveConsumers;
	private final OOCFuture<Void> _completion;
	private final AtomicBoolean _done;

	public OOCStreamMaterializer(MaterializedStore<IndexedMatrixValue> store, ToIntFunction<MatrixIndexes> linearize,
		MemoryAllowance allowance) {
		this(store, linearize, allowance, List.of());
	}

	public OOCStreamMaterializer(MaterializedStore<IndexedMatrixValue> store, ToIntFunction<MatrixIndexes> linearize,
		MemoryAllowance allowance, List<Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>> liveConsumers) {
		_store = store;
		_linearize = linearize;
		_allowance = allowance;
		_liveConsumers = List.copyOf(liveConsumers);
		_completion = new OOCFuture<>();
		_done = new AtomicBoolean(false);
	}

	public void attach(OOCStream<IndexedMatrixValue> source) {
		source.setSubscriber(this);
	}

	public OOCFuture<Void> completion() {
		return _completion;
	}

	@Override
	public void accept(OOCStream.QueueCallback<IndexedMatrixValue> callback) {
		if(_done.get()) {
			callback.close();
			return;
		}
		try(callback) {
			if(callback.isFailure()) {
				try {
					callback.get();
				}
				catch(DMLRuntimeException ex) {
					fail(ex);
				}
				return;
			}
			if(callback.isEos()) {
				finish();
				return;
			}
			publish(callback);
		}
		catch(RuntimeException ex) {
			fail(DMLRuntimeException.of(ex));
		}
	}

	private void publish(OOCStream.QueueCallback<IndexedMatrixValue> callback) {
		IndexedMatrixValue value = callback.get();
		int index = _linearize.applyAsInt(value.getIndexes());
		StoreLease<IndexedMatrixValue> lease;
		if(callback instanceof InMemoryQueueCallback managed && managed.getManagedBytes() > 0) {
			lease = _store.publishPinnedLive(index, managed.extractManagedPayload());
		}
		else {
			// To handle not-yet managed callbacks
			long bytes = serializedSize(value);
			_allowance.reserveBlocking(bytes);
			lease = _store.publishPinnedLive(index, value, bytes, _allowance);
		}
		try(lease) {
			for(Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> liveConsumer : _liveConsumers) {
				try(OOCStream.QueueCallback<IndexedMatrixValue> alias = new MaterializedCallback(lease.retain())) {
					liveConsumer.accept(alias);
				}
			}
		}
	}

	private void finish() {
		if(!_done.compareAndSet(false, true))
			return;
		try {
			_store.complete();
		}
		catch(RuntimeException ex) {
			deliverEos(DMLRuntimeException.of(ex));
			_completion.completeExceptionally(ex);
			return;
		}
		deliverEos(null);
		_completion.complete(null);
	}

	private void fail(DMLRuntimeException failure) {
		if(!_done.compareAndSet(false, true))
			return;
		deliverEos(failure);
		_completion.completeExceptionally(failure);
	}

	private void deliverEos(DMLRuntimeException failure) {
		for(Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> liveConsumer : _liveConsumers) {
			try {
				liveConsumer.accept(OOCStream.eos(failure));
			}
			catch(RuntimeException ignored) {
			}
		}
	}

	private static long serializedSize(IndexedMatrixValue value) {
		return ((MatrixBlock) value.getValue()).getExactSerializedSize();
	}
}
