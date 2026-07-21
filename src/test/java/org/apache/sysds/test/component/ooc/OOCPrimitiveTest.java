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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.primitives.OOCPrimitive;
import org.apache.sysds.runtime.ooc.stream.FilteredOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.junit.Assert;
import org.junit.Test;

public class OOCPrimitiveTest {
	@Test
	public void testRetainForgottenCacheCallback() {
		OOCCacheManager.reset();
		try(OOCStream.QueueCallback<IndexedMatrixValue> callback = OOCCacheManager.putAndPin(1, 1,
			new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, 7d)))) {
			OOCCacheManager.forget(1, 1);
			try(OOCStream.QueueCallback<IndexedMatrixValue> retained = callback.keepOpen()) {
				Assert.assertEquals(7, retained.get().getValue().get(0, 0), 0);
			}
		}
		finally {
			OOCCacheManager.reset();
		}
	}

	@Test
	public void testGraphPatternsAndExecution() {
		TestPrimitive source = new TestPrimitive(List.of());
		TestPrimitive sink = new TestPrimitive(List.of(source, source));

		Assert.assertEquals(List.of(source), sink.getChildren());
		Assert.assertEquals(List.of(sink), source.getParents());
		source.inferPatterns();
		Assert.assertEquals(OOCAccessPattern.ANY, source.getAccessPattern());
		Assert.assertEquals(OOCAccessPattern.ANY, sink.getAccessPattern());
		Assert.assertEquals(OOCAccessPattern.COL_MAJOR, OOCAccessPattern.ROW_MAJOR.transposed());
		Assert.assertEquals(OOCAccessPattern.UNKNOWN, OOCAccessPattern.ROW_MAJOR.fused(OOCAccessPattern.COL_MAJOR));

		SubscribableTaskQueue<Integer> stream = new SubscribableTaskQueue<>();
		stream.assignPrimitive(sink);
		FilteredOOCStream<Integer> filtered = new FilteredOOCStream<>(stream, ignored -> true);
		Assert.assertSame(sink, filtered.getPrimitive());
		stream.start();
		filtered.start();
		Assert.assertTrue(sink.hasStartedExecution());
		Assert.assertEquals(1, sink._executions);
		Assert.assertEquals(1, source._executions);
		Assert.assertEquals(OOCAccessPattern.ROW_MAJOR, sink.getAccessPattern());
		sink.inferPatterns();
		sink.requestPattern(OOCAccessPattern.COL_MAJOR);
		Assert.assertEquals(OOCAccessPattern.ROW_MAJOR, sink.getAccessPattern());
	}

	@Test
	public void testDataGenMapTransposePipeline() {
		SubscribableTaskQueue<IndexedMatrixValue> generated = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> mapped = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> transposed = new SubscribableTaskQueue<>();
		generated.setData(new MatrixObject(ValueType.FP64, "/dev/null",
			new MetaDataFormat(new MatrixCharacteristics(2, 3, 1), FileFormat.BINARY)));
		mapped.setData(new MatrixObject(ValueType.FP64, "/dev/null",
			new MetaDataFormat(new MatrixCharacteristics(2, 3, 1), FileFormat.BINARY)));
		transposed.setData(new MatrixObject(ValueType.FP64, "/dev/null",
			new MetaDataFormat(new MatrixCharacteristics(3, 2, 1), FileFormat.BINARY)));

		OOCInstructionUtils.dataGen(generated,
			indexes -> new MatrixBlock(1, 1, (double) indexes.getRowIndex() * 10 + indexes.getColumnIndex()),
			new StreamContext());
		OOCInstructionUtils.equiMapBlock(generated, mapped, input -> new MatrixBlock(1, 1, input.get(0, 0) + 1),
			new StreamContext());
		OOCInstructionUtils.transpose(mapped, transposed, new StreamContext());

		transposed.start();
		Map<String, Double> values = new HashMap<>();
		OOCStream.QueueCallback<IndexedMatrixValue> callback;
		while((callback = transposed.dequeueCB()) != null) {
			try(OOCStream.QueueCallback<IndexedMatrixValue> current = callback) {
				IndexedMatrixValue value = current.get();
				values.put(value.getIndexes().getRowIndex() + "," + value.getIndexes().getColumnIndex(),
					value.getValue().get(0, 0));
			}
		}
		Assert.assertEquals(Map.of("1,1", 12.0, "2,1", 13.0, "3,1", 14.0, "1,2", 22.0, "2,2", 23.0, "3,2", 24.0),
			values);
	}

	@Test
	public void testJoinOutOfOrder() {
		SubscribableTaskQueue<IndexedMatrixValue> left = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> right = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> joined = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> addends = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> output = new SubscribableTaskQueue<>();
		for(SubscribableTaskQueue<IndexedMatrixValue> stream : List.of(left, right, joined, addends, output))
			stream.setData(new MatrixObject(ValueType.FP64, "/dev/null",
				new MetaDataFormat(new MatrixCharacteristics(1, 2, 1), FileFormat.BINARY)));
		CachingStream cachedLeft = new CachingStream(left);
		left.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, 10d)));
		left.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 2), new MatrixBlock(1, 1, 20d)));
		right.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 2), new MatrixBlock(1, 1, 2d)));
		right.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, 1d)));
		addends.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1, 1, 100d)));
		addends.enqueue(new IndexedMatrixValue(new MatrixIndexes(1, 2), new MatrixBlock(1, 1, 200d)));
		left.closeInput();
		right.closeInput();
		addends.closeInput();
		OOCInstructionUtils.equiJoin(cachedLeft, right, joined,
			(l, r) -> new MatrixBlock(1, 1, l.get(0, 0) + r.get(0, 0)), new StreamContext());
		OOCInstructionUtils.equiJoin(joined, addends, output,
			(l, r) -> new MatrixBlock(1, 1, l.get(0, 0) + r.get(0, 0)), new StreamContext());

		output.start();
		Map<Long, Double> values = new HashMap<>();
		OOCStream.QueueCallback<IndexedMatrixValue> callback;
		while((callback = output.dequeueCB()) != null)
			try(OOCStream.QueueCallback<IndexedMatrixValue> current = callback) {
				values.put(current.get().getIndexes().getColumnIndex(), current.get().getValue().get(0, 0));
			}
		Assert.assertEquals(Map.of(1L, 111.0, 2L, 222.0), values);
		cachedLeft.scheduleDeletion();
	}

	private static final class TestPrimitive extends OOCPrimitive {
		private int _executions;

		private TestPrimitive(List<OOCPrimitive> children) {
			super(null, children);
		}

		@Override
		protected void startExecution() {
			_executions++;
			onComplete();
		}

		@Override
		protected void inferPatternsInternal() {
			_pattern = OOCAccessPattern.ANY;
			inferParentPatterns();
		}

		@Override
		protected void requestPatternInternal(OOCAccessPattern accessPattern) {
			_pattern = accessPattern;
		}
	}
}
