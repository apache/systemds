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

package org.apache.sysds.test.functions.io.delta;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.HashMap;

import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * End-to-end DML test of the native Delta <b>frame</b> read/write path.
 *
 * <p>As in the matrix variant, the write and the read run as two separate
 * SystemDS executions so the read is a genuine disk read rather than an
 * in-memory cache hit. We additionally assert via {@link CacheStatistics} that
 * the write run wrote (delta + text reference) and the read run read (delta +
 * text reference) from HDFS, so a short-circuited path would fail the test.</p>
 */
public class FrameDeltaReadWriteTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/io/delta/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameDeltaReadWriteTest.class.getSimpleName() + "/";
	private final static String WRITE_NAME = "FrameDeltaWrite";
	private final static String READ_NAME = "FrameDeltaReadCompare";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(WRITE_NAME,
			new TestConfiguration(TEST_CLASS_DIR, WRITE_NAME, new String[] { "ref" }));
		addTestConfiguration(READ_NAME,
			new TestConfiguration(TEST_CLASS_DIR, READ_NAME, new String[] { "R" }));
	}

	@Test
	public void testDenseRoundTrip() {
		runFrameDeltaRoundTrip(200, 12, 1.0);
	}

	@Test
	public void testSparseRoundTrip() {
		runFrameDeltaRoundTrip(640, 8, 0.2);
	}

	@Test
	public void testMultiBatchRoundTrip() {
		runFrameDeltaRoundTrip(9000, 4, 1.0);
	}

	private void runFrameDeltaRoundTrip(int rows, int cols, double sparsity) {
		try {
			String HOME = SCRIPT_DIR + TEST_DIR;

			// ---- phase 1: write the frame as a Delta table + text reference ----
			getAndLoadTestConfiguration(WRITE_NAME);
			String deltaPath = output("deltaTable");
			String refPath = output("ref");
			fullDMLScriptName = HOME + WRITE_NAME + ".dml";
			programArgs = new String[] { "-stats", "-args",
				String.valueOf(rows), String.valueOf(cols), String.valueOf(sparsity),
				deltaPath, refPath };
			runTest(true, false, null, -1);

			//the write run must materialize two objects to disk: the frame Delta
			//table under test + the matrix text reference. FrameWriterDelta genuinely
			//hitting HDFS is what produces the frame-side write statistic.
			long hdfsWrites = CacheStatistics.getHDFSWrites();
			assertTrue("expected >= 2 HDFS writes in the write run (delta frame + reference), got "
				+ hdfsWrites, hdfsWrites >= 2);
			//and a real Delta table (transaction log) must have been created
			assertTrue("missing Delta transaction log under " + deltaPath,
				new File(deltaPath, "_delta_log").isDirectory());

			// ---- phase 2: fresh execution reads the Delta frame and compares ----
			getAndLoadTestConfiguration(READ_NAME);
			fullDMLScriptName = HOME + READ_NAME + ".dml";
			programArgs = new String[] { "-stats", "-args",
				deltaPath, refPath, output("R") };
			runTest(true, false, null, -1);

			long hdfsReads = CacheStatistics.getHDFSHits();
			assertTrue("expected >= 2 HDFS reads in the read run (delta + reference), got "
				+ hdfsReads, hdfsReads >= 2);

			HashMap<CellIndex, Double> R = readDMLMatrixFromOutputDir("R");
			double diff = R.getOrDefault(new CellIndex(1, 1), 0.0);
			double nrow = R.getOrDefault(new CellIndex(1, 2), 0.0);
			double ncol = R.getOrDefault(new CellIndex(1, 3), 0.0);

			assertEquals("reconstruction error", 0.0, diff, 1e-12);
			assertEquals("discovered rows", rows, (int) nrow);
			assertEquals("discovered cols", cols, (int) ncol);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
