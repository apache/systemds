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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCInstruction;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCIOHandler;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

import java.util.concurrent.ConcurrentHashMap;
import java.util.List;
import java.util.concurrent.locks.LockSupport;

public class SourceOOCStream extends SubscribableTaskQueue<IndexedMatrixValue> {
	private final ConcurrentHashMap<MatrixIndexes, OOCIOHandler.SourceBlockDescriptor> _idx;
	private static final long BACKPRESSURE_PARK_NANOS = 1_000_000L;
	private static final long MAX_BACKPRESSURE_PARK_NANOS = 200_000_000L;

	public SourceOOCStream() {
		this._idx = new ConcurrentHashMap<>();
	}

	public void enqueue(IndexedMatrixValue value, OOCIOHandler.SourceBlockDescriptor descriptor) {
		if(descriptor == null)
			throw new IllegalArgumentException("Source descriptor must not be null");
		waitForBackpressure();
		MatrixIndexes key = new MatrixIndexes(descriptor.indexes);
		_idx.put(key, descriptor);
		super.enqueue(value);
	}

	public void enqueueGroup(List<IndexedMatrixValue> values, OOCIOHandler.GroupSourceBlockDescriptor descriptor) {
		if (descriptor == null)
			throw new IllegalArgumentException("Group source descriptor must not be null");
		if (values == null || values.isEmpty())
			return;
		waitForBackpressure();
		boolean delivered = tryDeliverCallback(new SourceGroupCallback(values, descriptor), values.size());
		if (!delivered) {
			// Fallback to individual enqueues if no subscriber yet
			for (int i = 0; i < values.size(); i++) {
				OOCIOHandler.SourceBlockDescriptor d = descriptor.blocks.get(i);
				enqueue(values.get(i), d);
			}
		}
	}

	@Override
	public void enqueue(IndexedMatrixValue val) {
		throw new UnsupportedOperationException("Use enqueue(value, descriptor) for source streams");
	}

	public OOCIOHandler.SourceBlockDescriptor getDescriptor(MatrixIndexes indexes) {
		return _idx.get(indexes);
	}

	private void waitForBackpressure() {
		int limit = OOCInstruction.getComputeBackpressureThreshold();
		if(limit <= 0)
			return;
		long parkNanos = BACKPRESSURE_PARK_NANOS;
		while(!OOCCacheManager.canClaimMemory()) {
			LockSupport.parkNanos(parkNanos);
			if (Thread.interrupted())
				throw new DMLRuntimeException(new InterruptedException());
			if (parkNanos < MAX_BACKPRESSURE_PARK_NANOS)
				parkNanos = Math.min(parkNanos * 2, MAX_BACKPRESSURE_PARK_NANOS);
		}
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		if(msg.isCancelled())
			return;
		super.messageUpstream(msg);
	}

	public static class SourceGroupCallback implements OOCStream.GroupQueueCallback<IndexedMatrixValue> {
		private final List<IndexedMatrixValue> _data;
		private final OOCIOHandler.GroupSourceBlockDescriptor _descriptor;
		private DMLRuntimeException _failure;

		SourceGroupCallback(List<IndexedMatrixValue> data, OOCIOHandler.GroupSourceBlockDescriptor descriptor) {
			_data = data;
			_descriptor = descriptor;
		}

		public OOCIOHandler.GroupSourceBlockDescriptor getDescriptor() {
			return _descriptor;
		}

		@Override
		public int size() {
			return _data.size();
		}

		@Override
		public OOCStream.QueueCallback<IndexedMatrixValue> getCallback(int idx) {
			return new OOCStream.SimpleQueueCallback<>(_data.get(idx), _failure);
		}

		@Override
		public IndexedMatrixValue get() {
			throw new UnsupportedOperationException();
		}

		@Override
		public OOCStream.QueueCallback<IndexedMatrixValue> keepOpen() {
			return this;
		}

		@Override
		public void close() {
		}

		@Override
		public void fail(DMLRuntimeException failure) {
			_failure = failure;
		}

		@Override
		public boolean isEos() {
			return false;
		}

		@Override
		public boolean isFailure() {
			return _failure != null;
		}
	}
}
