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
 * End-to-end DML test of the native Delta read/write path.
 *
 * <p>The write and the read are run as two <b>separate</b> SystemDS executions
 * on purpose. If they shared a single script/process, SystemDS would reuse the
 * still-materialized in-memory matrix for the subsequent read and never invoke
 * {@link org.apache.sysds.runtime.io.ReaderDelta} at all (verified: the cache
 * reports 0 HDFS hits in that case). Splitting the executions forces a genuine
 * read from disk, and we additionally assert via {@link CacheStatistics} that
 * the read run actually performed HDFS reads (the Delta table + the text
 * reference) rather than serving the matrix from cache.</p>
 */
public class DeltaReadWriteTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/io/delta/";
	private final static String TEST_CLASS_DIR = TEST_DIR + DeltaReadWriteTest.class.getSimpleName() + "/";
	private final static String WRITE_NAME = "DeltaWrite";
	private final static String READ_NAME = "DeltaReadCompare";

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
		runDeltaRoundTrip(200, 12, 1.0);
	}

	@Test
	public void testSparseRoundTrip() {
		runDeltaRoundTrip(640, 8, 0.2);
	}

	@Test
	public void testMultiBatchRoundTrip() {
		runDeltaRoundTrip(9000, 4, 1.0);
	}

	private void runDeltaRoundTrip(int rows, int cols, double sparsity) {
		try {
			String HOME = SCRIPT_DIR + TEST_DIR;

			// ---- phase 1: write the matrix as a Delta table + text reference ----
			getAndLoadTestConfiguration(WRITE_NAME);
			String deltaPath = output("deltaTable");
			String refPath = output("ref");
			fullDMLScriptName = HOME + WRITE_NAME + ".dml";
			programArgs = new String[] { "-stats", "-args",
				String.valueOf(rows), String.valueOf(cols), String.valueOf(sparsity),
				deltaPath, refPath };
			runTest(true, false, null, -1);

			// the write run must have materialized two matrices to disk (the Delta
			// table under test + the text reference); WriterDelta genuinely hitting
			// HDFS is what produces these write-side cache statistics.
			long hdfsWrites = CacheStatistics.getHDFSWrites();
			assertTrue("expected >= 2 HDFS writes in the write run (delta + reference), got "
				+ hdfsWrites, hdfsWrites >= 2);
			// and a real Delta table (transaction log) must have been created
			assertTrue("missing Delta transaction log under " + deltaPath,
				new File(deltaPath, "_delta_log").isDirectory());

			// ---- phase 2: fresh execution reads the Delta table and compares ----
			getAndLoadTestConfiguration(READ_NAME);
			fullDMLScriptName = HOME + READ_NAME + ".dml";
			programArgs = new String[] { "-stats", "-args",
				deltaPath, refPath, output("R") };
			runTest(true, false, null, -1);

			// the read run must have materialized two matrices from disk (the Delta
			// table under test + the text reference); a cached/short-circuited read
			// would report fewer HDFS hits and fail here.
			long hdfsReads = CacheStatistics.getHDFSHits();
			assertTrue("expected >= 2 HDFS reads in the read run (delta + reference), got "
				+ hdfsReads, hdfsReads >= 2);

			HashMap<CellIndex, Double> R = readDMLMatrixFromOutputDir("R");
			//text-cell output omits exact zeros, so a missing cell means 0.0
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
