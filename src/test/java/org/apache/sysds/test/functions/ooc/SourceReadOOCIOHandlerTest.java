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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.ooc.cache.OOCIOHandler;
import org.apache.sysds.runtime.ooc.cache.OOCMatrixIOHandler;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class SourceReadOOCIOHandlerTest extends AutomatedTestBase {
	private static final String TEST_NAME = "SourceReadOOCIOHandler";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SourceReadOOCIOHandlerTest.class.getSimpleName() + "/";

	private OOCMatrixIOHandler handler;

	@Override
	@Before
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
		handler = new OOCMatrixIOHandler();
	}

	@After
	public void tearDown() {
		if (handler != null)
			handler.shutdown();
	}

	@Test
	public void testSourceReadCompletes() throws Exception {
		getAndLoadTestConfiguration(TEST_NAME);
		final int rows = 4;
		final int cols = 4;
		final int blen = 2;

		MatrixBlock src = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 7);
		String fname = input("binary_full");
		writeBinaryMatrix(src, fname, blen);

		SubscribableTaskQueue<IndexedMatrixValue> target = new SubscribableTaskQueue<>();
		OOCIOHandler.SourceReadRequest req = new OOCIOHandler.SourceReadRequest(fname, Types.FileFormat.BINARY,
			rows, cols, blen, src.getNonZeros(), Long.MAX_VALUE, true, target);

		OOCIOHandler.SourceReadResult res = handler.scheduleSourceRead(req).get();
		// Drain after EOF
		MatrixBlock reconstructed = drainToMatrix(target, rows, cols, blen);

		TestUtils.compareMatrices(src, reconstructed, 1e-12);
		org.junit.Assert.assertTrue(res.eof);
		org.junit.Assert.assertNull(res.continuation);
		org.junit.Assert.assertNotNull(res.blocks);
		org.junit.Assert.assertEquals((rows / blen) * (cols / blen), res.blocks.size());
		org.junit.Assert.assertTrue(res.blocks.stream().allMatch(b -> b.indexes != null));
	}

	@Test
	public void testSourceReadStopsOnBudgetAndContinues() throws Exception {
		getAndLoadTestConfiguration(TEST_NAME);
		final int rows = 4;
		final int cols = 4;
		final int blen = 2;

		MatrixBlock src = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 13);
		String fname = input("binary_budget");
		writeBinaryMatrix(src, fname, blen);

		long singleBlockSize = new MatrixBlock(blen, blen, false).getExactSerializedSize();
		long budget = singleBlockSize + 1; // ensure we stop before the second block

		SubscribableTaskQueue<IndexedMatrixValue> target = new SubscribableTaskQueue<>();
		OOCIOHandler.SourceReadRequest req = new OOCIOHandler.SourceReadRequest(fname, Types.FileFormat.BINARY,
			rows, cols, blen, src.getNonZeros(), budget, true, target);

		OOCIOHandler.SourceReadResult first = handler.scheduleSourceRead(req).get();
		org.junit.Assert.assertFalse(first.eof);
		org.junit.Assert.assertNotNull(first.continuation);
		org.junit.Assert.assertNotNull(first.blocks);

		OOCIOHandler.SourceReadResult second = handler.continueSourceRead(first.continuation, Long.MAX_VALUE).get();
		org.junit.Assert.assertTrue(second.eof);
		org.junit.Assert.assertNull(second.continuation);
		org.junit.Assert.assertNotNull(second.blocks);
		org.junit.Assert.assertEquals((rows / blen) * (cols / blen), first.blocks.size() + second.blocks.size());

		MatrixBlock reconstructed = drainToMatrix(target, rows, cols, blen);
		TestUtils.compareMatrices(src, reconstructed, 1e-12);
	}

	private void writeBinaryMatrix(MatrixBlock mb, String fname, int blen) throws Exception {
		MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
		writer.writeMatrixToHDFS(mb, fname, mb.getNumRows(), mb.getNumColumns(), blen, mb.getNonZeros());
	}

	private MatrixBlock drainToMatrix(SubscribableTaskQueue<IndexedMatrixValue> target, int rows, int cols, int blen) {
		List<IndexedMatrixValue> blocks = new ArrayList<>();
		IndexedMatrixValue tmp;
		while((tmp = target.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
			blocks.add(tmp);
		}

		MatrixBlock out = new MatrixBlock(rows, cols, false);
		for (IndexedMatrixValue imv : blocks) {
			int rowOffset = (int)((imv.getIndexes().getRowIndex() - 1) * blen);
			int colOffset = (int)((imv.getIndexes().getColumnIndex() - 1) * blen);
			((MatrixBlock)imv.getValue()).putInto(out, rowOffset, colOffset, true);
		}
		out.recomputeNonZeros();
		return out;
	}
}
