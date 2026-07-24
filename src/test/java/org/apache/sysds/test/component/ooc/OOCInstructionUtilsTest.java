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

package org.apache.sysds.test.component.ooc;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.store.MaterializedCallback;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.junit.Assert;
import org.junit.Test;

public class OOCInstructionUtilsTest {
	@Test
	public void testSubmitTasksClosesCallbacksAfterCompletion() throws Exception {
		SubscribableTaskQueue<IndexedMatrixValue> source = new SubscribableTaskQueue<>();
		AtomicInteger processed = new AtomicInteger();
		AtomicInteger released = new AtomicInteger();
		CompletableFuture<Void> completion = OOCInstructionUtils.submitOOCTasks(source, callback -> {
			Assert.assertEquals(1, callback.get().getIndexes().getRowIndex());
			processed.incrementAndGet();
		}, new StreamContext().addOutStream());

		IndexedMatrixValue value = new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, 1.0));
		source.enqueue(new MaterializedCallback<>(new StoreLease<>(value, released::incrementAndGet)));
		source.closeInput();
		completion.get(10, TimeUnit.SECONDS);

		Assert.assertEquals(1, processed.get());
		Assert.assertEquals(1, released.get());
	}

	@Test
	public void testSubmitTasksWaitsForAllStreams() throws Exception {
		SubscribableTaskQueue<Integer> first = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<Integer> second = new SubscribableTaskQueue<>();
		AtomicInteger processed = new AtomicInteger();
		CompletableFuture<Void> completion = OOCInstructionUtils.submitOOCTasks(List.of(first, second),
			(index, callback) -> processed.addAndGet(callback.get()), new StreamContext().addOutStream());

		first.enqueue(1);
		first.closeInput();
		second.enqueue(2);
		Assert.assertFalse(completion.isDone());
		second.closeInput();
		completion.get(10, TimeUnit.SECONDS);
		Assert.assertEquals(3, processed.get());
	}

	@Test
	public void testSubmitTaskPropagatesFailure() throws Exception {
		SubscribableTaskQueue<Integer> output = new SubscribableTaskQueue<>();
		AtomicReference<DMLRuntimeException> propagated = new AtomicReference<>();
		output.setSubscriber(callback -> {
			try(callback) {
				if(callback.isFailure()) {
					try {
						callback.get();
					}
					catch(DMLRuntimeException failure) {
						propagated.compareAndSet(null, failure);
					}
				}
			}
		});

		OOCFuture<Void> completion = OOCInstructionUtils.submitOOCTask(() -> {
			throw new DMLRuntimeException("injected failure");
		}, new StreamContext().addOutStream(output));
		try {
			completion.get(10, TimeUnit.SECONDS);
			Assert.fail("Expected task failure");
		}
		catch(ExecutionException expected) {
			Assert.assertTrue(expected.getCause() instanceof DMLRuntimeException);
		}
		output.closeInput();
		Assert.assertNotNull(propagated.get());
		Assert.assertEquals("injected failure", propagated.get().getMessage());
	}
}
