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

package org.apache.sysds.runtime.ooc.cache;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SourceBackedCacheSchedulerTest extends AutomatedTestBase {
	private static final String TEST_NAME = "SourceBackedCacheScheduler";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SourceBackedCacheSchedulerTest.class.getSimpleName() + "/";

	private OOCMatrixIOHandler handler;
	private OOCLRUCacheScheduler scheduler;

	@Override
	@Before
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
		handler = new OOCMatrixIOHandler();
		scheduler = new OOCLRUCacheScheduler(handler, 0, Long.MAX_VALUE);
	}

	@After
	public void tearDown() {
		if (scheduler != null)
			scheduler.shutdown();
		if (handler != null)
			handler.shutdown();
	}

	@Test
	public void testPutSourceBackedAndReload() throws Exception {
		getAndLoadTestConfiguration(TEST_NAME);
		final int rows = 4;
		final int cols = 4;
		final int blen = 2;

		MatrixBlock src = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 23);
		String fname = input("binary_src_cache");
		writeBinaryMatrix(src, fname, blen);

		SubscribableTaskQueue<IndexedMatrixValue> target = new SubscribableTaskQueue<>();
		OOCIOHandler.SourceReadRequest req = new OOCIOHandler.SourceReadRequest(fname, Types.FileFormat.BINARY,
			rows, cols, blen, src.getNonZeros(), Long.MAX_VALUE, true, target);

		OOCIOHandler.SourceReadResult res = handler.scheduleSourceRead(req).get();
		IndexedMatrixValue imv = target.dequeue();
		OOCIOHandler.SourceBlockDescriptor desc = res.blocks.get(0);

		BlockKey key = new BlockKey(11, 0);
		BlockEntry entry = scheduler.putAndPinSourceBacked(key, imv,
			((MatrixBlock) imv.getValue()).getExactSerializedSize(), desc);
		org.junit.Assert.assertEquals(BlockState.WARM, entry.getState());

		scheduler.unpin(entry);
		org.junit.Assert.assertEquals(BlockState.COLD, entry.getState());
		org.junit.Assert.assertNull(entry.getDataUnsafe());

		BlockEntry reloaded = scheduler.request(key).get();
		IndexedMatrixValue reloadImv = (IndexedMatrixValue) reloaded.getData();
		MatrixBlock expected = expectedBlock(src, desc.indexes, blen);
		TestUtils.compareMatrices(expected, (MatrixBlock) reloadImv.getValue(), 1e-12);
	}

	private void writeBinaryMatrix(MatrixBlock mb, String fname, int blen) throws Exception {
		MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
		writer.writeMatrixToHDFS(mb, fname, mb.getNumRows(), mb.getNumColumns(), blen, mb.getNonZeros());
	}

	private MatrixBlock expectedBlock(MatrixBlock src, org.apache.sysds.runtime.matrix.data.MatrixIndexes idx, int blen) {
		int rowStart = (int) ((idx.getRowIndex() - 1) * blen);
		int colStart = (int) ((idx.getColumnIndex() - 1) * blen);
		int rowEnd = Math.min(rowStart + blen - 1, src.getNumRows() - 1);
		int colEnd = Math.min(colStart + blen - 1, src.getNumColumns() - 1);
		return src.slice(rowStart, rowEnd, colStart, colEnd);
	}
}
