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

package org.apache.sysds.test.functions.ooc;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCInstruction;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

public class SplitMergeOOCStreamTest extends AutomatedTestBase {
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSplitMergeNonCached() {
		PrimitiveHarness h = new PrimitiveHarness();
		SubscribableTaskQueue<IndexedMatrixValue> src = new SubscribableTaskQueue<>();
		List<IndexedMatrixValue> input = createInput(24);

		List<OOCStream<IndexedMatrixValue>> splits = h.split(src, partitionByRow(3), 3);
		OOCStream<IndexedMatrixValue> merged = h.merge(splits);

		Assert.assertFalse("Merged stream should not expose cache for non-cached inputs", merged.hasStreamCache());
		feed(src, input);
		Map<Long, IndexedMatrixValue> outByKey = h.collect(merged);
		assertSameItems(outByKey, input);
	}

	@Test
	public void testSplitMergeCached() {
		PrimitiveHarness h = new PrimitiveHarness();
		SubscribableTaskQueue<IndexedMatrixValue> base = new SubscribableTaskQueue<>();
		CachingStream cache = new CachingStream(base);
		List<IndexedMatrixValue> input = createInput(24);

		feed(base, input);

		OOCStream<IndexedMatrixValue> cachedRead = cache.getReadStream();
		List<OOCStream<IndexedMatrixValue>> splits = h.split(cachedRead, partitionByRow(4), 4);
		OOCStream<IndexedMatrixValue> merged = h.merge(splits);

		Assert.assertTrue("Merged stream should expose cache when all inputs share one cache", merged.hasStreamCache());
		Assert.assertSame("Merged stream should expose the shared cache", cache, merged.getStreamCache());
		Map<Long, IndexedMatrixValue> outByKey = h.collect(merged);
		assertSameItems(outByKey, input);
	}

	@Test
	public void testSplitMergeCbind1500x3000() {
		PrimitiveHarness h = new PrimitiveHarness();
		SubscribableTaskQueue<IndexedMatrixValue> leftSrc = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> rightSrc = new SubscribableTaskQueue<>();

		// Two logical 1500x1500 matrices tiled into 1k x 1k blocks.
		List<IndexedMatrixValue> leftInput = createTiled1500Input(11, 12, 21, 22);
		List<IndexedMatrixValue> rightInput = createTiled1500Input(111, 112, 121, 122);

		List<OOCStream<IndexedMatrixValue>> leftByRow = h.split(leftSrc, imv -> (int) (imv.getIndexes().getRowIndex() - 1), 2);
		List<OOCStream<IndexedMatrixValue>> rightByRow = h.split(rightSrc, imv -> (int) (imv.getIndexes().getRowIndex() - 1), 2);

		feed(leftSrc, leftInput);
		feed(rightSrc, rightInput);

		OOCStream<IndexedMatrixValue> row1Out = buildCbindRowPartition(h, leftByRow.get(0), rightByRow.get(0));
		OOCStream<IndexedMatrixValue> row2Out = buildCbindRowPartition(h, leftByRow.get(1), rightByRow.get(1));
		OOCStream<IndexedMatrixValue> merged = h.merge(List.of(row1Out, row2Out));
		MatrixObject outMo = createStreamBackedMatrixObject(merged, 1500, 3000, 1000);
		MatrixBlock oocCbind = outMo.acquireReadAndRelease();

		MatrixBlock leftCp = materializeThroughMatrixObject(leftInput, 1500, 1500, 1000);
		MatrixBlock rightCp = materializeThroughMatrixObject(rightInput, 1500, 1500, 1000);
		MatrixBlock cpCbind = leftCp.append(rightCp);
		TestUtils.compareMatrices(cpCbind, oocCbind, 0.0, "OOC cbind result differs from CP cbind result");
	}

	private static Function<IndexedMatrixValue, Integer> partitionByRow(int numPartitions) {
		return imv -> (int)Math.floorMod(imv.getIndexes().getRowIndex(), numPartitions);
	}

	private static void feed(SubscribableTaskQueue<IndexedMatrixValue> src, List<IndexedMatrixValue> input) {
		for(IndexedMatrixValue imv : input)
			src.enqueue(imv);
		src.closeInput();
	}

	private static List<IndexedMatrixValue> createInput(int n) {
		List<IndexedMatrixValue> input = new ArrayList<>(n);
		for(int i = 1; i <= n; i++)
			input.add(new IndexedMatrixValue(new MatrixIndexes(i, 1), new MatrixBlock(1, 1, (double)i)));
		return input;
	}

	private static List<IndexedMatrixValue> createTiled1500Input(double v11, double v12, double v21, double v22) {
		return List.of(
			new IndexedMatrixValue(new MatrixIndexes(1, 1), new MatrixBlock(1000, 1000, v11)),
			new IndexedMatrixValue(new MatrixIndexes(1, 2), new MatrixBlock(1000, 500, v12)),
			new IndexedMatrixValue(new MatrixIndexes(2, 1), new MatrixBlock(500, 1000, v21)),
			new IndexedMatrixValue(new MatrixIndexes(2, 2), new MatrixBlock(500, 500, v22))
		);
	}

	private OOCStream<IndexedMatrixValue> buildCbindRowPartition(PrimitiveHarness h,
		OOCStream<IndexedMatrixValue> leftPart, OOCStream<IndexedMatrixValue> rightPart) {
		List<OOCStream<IndexedMatrixValue>> leftByCol = h.split(leftPart, imv -> (int) (imv.getIndexes().getColumnIndex() - 1), 2);
		List<OOCStream<IndexedMatrixValue>> rightByCol = h.split(rightPart, imv -> (int) (imv.getIndexes().getColumnIndex() - 1), 2);

		OOCStream<IndexedMatrixValue> leftC1 = leftByCol.get(0);
		OOCStream<IndexedMatrixValue> leftC2 = leftByCol.get(1);
		OOCStream<IndexedMatrixValue> rightC1 = rightByCol.get(0);
		OOCStream<IndexedMatrixValue> rightC2 = rightByCol.get(1);

		CachingStream rightC1Cache = h.cache(rightC1);
		OOCStream<IndexedMatrixValue> rightC1ForCritical = rightC1Cache.getReadStream();
		OOCStream<IndexedMatrixValue> rightC1ForTail = rightC1Cache.getReadStream();

		SubscribableTaskQueue<IndexedMatrixValue> outCol1 = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> outCol2 = new SubscribableTaskQueue<>();
		SubscribableTaskQueue<IndexedMatrixValue> outCol3 = new SubscribableTaskQueue<>();
		Function<IndexedMatrixValue, MatrixIndexes> rowKey = imv -> new MatrixIndexes(imv.getIndexes().getRowIndex(), 1);

		CompletableFuture<Void> f1 = h.map(leftC1, outCol1, imv -> new IndexedMatrixValue(
			new MatrixIndexes(imv.getIndexes().getRowIndex(), 1), imv.getValue()));
		CompletableFuture<Void> f2 = h.join(leftC2, rightC1ForCritical, outCol2, (left, right) -> {
			MatrixBlock lb = (MatrixBlock) left.getValue();
			MatrixBlock rb = (MatrixBlock) right.getValue();
			MatrixBlock critical = cbindBlocks(lb, sliceCols(rb, 0, 500));
			return new IndexedMatrixValue(new MatrixIndexes(left.getIndexes().getRowIndex(), 2), critical);
		}, rowKey);
		CompletableFuture<Void> f3 = h.join(rightC1ForTail, rightC2, outCol3, (r1, r2) -> {
			MatrixBlock rb1 = (MatrixBlock) r1.getValue();
			MatrixBlock rb2 = (MatrixBlock) r2.getValue();
			MatrixBlock tail = cbindBlocks(sliceCols(rb1, 500, 1000), rb2);
			return new IndexedMatrixValue(new MatrixIndexes(r1.getIndexes().getRowIndex(), 3), tail);
		}, rowKey);

		h.await(CompletableFuture.allOf(f1, f2, f3));
		return h.merge(List.of(outCol1, outCol2, outCol3));
	}

	private static MatrixBlock sliceCols(MatrixBlock in, int colStart, int colEndExclusive) {
		int rows = in.getNumRows();
		int cols = colEndExclusive - colStart;
		MatrixBlock out = new MatrixBlock(rows, cols, false);
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < cols; c++)
				out.set(r, c, in.get(r, colStart + c));
		}
		return out;
	}

	private static MatrixBlock cbindBlocks(MatrixBlock left, MatrixBlock right) {
		int rows = left.getNumRows();
		if(rows != right.getNumRows())
			throw new IllegalArgumentException("Row mismatch in cbindBlocks");
		int lCols = left.getNumColumns();
		int rCols = right.getNumColumns();
		MatrixBlock out = new MatrixBlock(rows, lCols + rCols, false);
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < lCols; c++)
				out.set(r, c, left.get(r, c));
			for(int c = 0; c < rCols; c++)
				out.set(r, lCols + c, right.get(r, c));
		}
		return out;
	}

	private static MatrixObject createStreamBackedMatrixObject(OOCStream<IndexedMatrixValue> stream, long rows,
		long cols, int blen) {
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blen, -1);
		MatrixObject mo = new MatrixObject(ValueType.FP64, null, new MetaDataFormat(mc, FileFormat.BINARY));
		mo.setStreamHandle(stream);
		return mo;
	}

	private static MatrixBlock materializeThroughMatrixObject(List<IndexedMatrixValue> blocks, int rows, int cols, int blen) {
		SubscribableTaskQueue<IndexedMatrixValue> src = new SubscribableTaskQueue<>();
		feed(src, blocks);
		return createStreamBackedMatrixObject(src, rows, cols, blen).acquireReadAndRelease();
	}

	private static long key(long row, long col) {
		return row * 1_000_000L + col;
	}

	private static long key(IndexedMatrixValue imv) {
		return key(imv.getIndexes().getRowIndex(), imv.getIndexes().getColumnIndex());
	}

	private static void assertSameItems(Map<Long, IndexedMatrixValue> outByKey, List<IndexedMatrixValue> expected) {
		Set<Long> expectedKeys = new HashSet<>(expected.size());
		for(IndexedMatrixValue imv : expected)
			expectedKeys.add(key(imv));

		Assert.assertEquals("Unexpected number of output blocks", expected.size(), outByKey.size());
		Assert.assertEquals("Output keys differ from input keys", expectedKeys, outByKey.keySet());
	}

	private static class PrimitiveHarness extends OOCInstruction {
		PrimitiveHarness() {
			super(OOCType.Tee, "split_merge_test", "split_merge_test");
		}

		@Override
		public void processInstruction(ExecutionContext ec) {}

		<T> List<OOCStream<T>> split(OOCStream<T> source, Function<T, Integer> partitionFunc, int numPartitions) {
			return splitOOCStream(source, partitionFunc, numPartitions);
		}

		<T> OOCStream<T> merge(List<OOCStream<T>> streams) {
			return mergeOOCStreams(streams);
		}

		CachingStream cache(OOCStream<IndexedMatrixValue> stream) {
			return new CachingStream(stream);
		}

		<R> CompletableFuture<Void> map(OOCStream<IndexedMatrixValue> in, OOCStream<R> out,
			Function<IndexedMatrixValue, R> mapper) {
			return mapOOC(in, out, mapper);
		}

		CompletableFuture<Void> join(OOCStream<IndexedMatrixValue> left, OOCStream<IndexedMatrixValue> right,
			OOCStream<IndexedMatrixValue> out,
			BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> mapper,
			Function<IndexedMatrixValue, MatrixIndexes> keyFn) {
			return joinOOC(left, right, out, mapper, keyFn);
		}

		Map<Long, IndexedMatrixValue> collect(OOCStream<IndexedMatrixValue> stream) {
			Map<Long, IndexedMatrixValue> out = new ConcurrentHashMap<>();
			await(collectToMap(stream, out));
			return out;
		}

		void await(CompletableFuture<Void> future) {
			try {
				future.join();
			}
			catch(CompletionException ex) {
				throw ex.getCause() instanceof RuntimeException ? (RuntimeException) ex.getCause() : ex;
			}
		}

		private CompletableFuture<Void> collectToMap(OOCStream<IndexedMatrixValue> stream, Map<Long, IndexedMatrixValue> out) {
			addInStream(stream);
			addOutStream();
			return submitOOCTasks(stream, cb -> {
				IndexedMatrixValue item = cb.get();
				long k = key(item);
				IndexedMatrixValue prev = out.putIfAbsent(k, item);
				Assert.assertNull("Duplicate output item for key " + k, prev);
			});
		}
	}
}
