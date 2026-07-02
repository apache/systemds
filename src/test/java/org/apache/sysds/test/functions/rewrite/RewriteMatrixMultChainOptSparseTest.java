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

package org.apache.sysds.test.functions.rewrite;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;

import org.junit.Assert;
import org.junit.runners.Parameterized;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class RewriteMatrixMultChainOptSparseTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteMatrixMultChainOpSparse";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteMatrixMultChainOptSparseTest.class.getSimpleName() + "/";
	private static final String PACKAGE = "org.apache.sysds.hops.rewrite.HopRewriteRule";
	private static Level _oldLevel;

	@Parameterized.Parameter(0)
	public int rows;

	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public double[] sparsities;

	@Parameterized.Parameter(3)
	public double eps;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {rows, cols, sparsities, eps},
			{1000, 300, new double[]{0.10d, 0.10d}, Math.pow(10, -10)},
			// {2, 300, new double[]{0.005, 1}, Math.pow(10, -10)},
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
		_oldLevel = Logger.getLogger(PACKAGE).getLevel();
		Logger.getLogger(PACKAGE).setLevel(Level.TRACE);
	}

	@Override
	public void tearDown() {
		super.tearDown();
		Logger.getLogger(PACKAGE).setLevel(_oldLevel);
	}

	@Test
	public void testMatrixMultChainOptSparseNoRewrites() {
		testRewriteMatrixMultChainOpSparse(false);
	}

	@Test
	public void testMatrixMultChainOptSparseRewrites() {
		testRewriteMatrixMultChainOpSparse(true);
	}

	private void testRewriteMatrixMultChainOpSparse(boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES;
		boolean oldFlag2 = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;

		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "hops", "-stats", "-args", input("X"), input("Y"), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = rewrites;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewrites;
			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsities[0], 7);
			double[][] Y = getRandomMatrix(cols, 1, -1, 1, sparsities[1], 3);
			long X_nnz = Stream.of(X).mapToLong(row -> DoubleStream.of(row).filter(val -> val != 0).count()).sum();
			long Y_nnz = Stream.of(Y).mapToLong(row -> DoubleStream.of(row).filter(val -> val != 0).count()).sum();
			writeInputMatrixWithMTD("X", X, X_nnz, true);
			writeInputMatrixWithMTD("Y", Y, Y_nnz, true);


			//execute tests
			TestAppender appender = LoggingUtils.overwrite(); // capture log output
			runTest(true, false, null, -1);
			List<LoggingEvent> log_out = LoggingUtils.reinsert(appender); // revert the logger to print to stdout

			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites) {
				Assert.assertTrue(log_out.stream().anyMatch(
					l -> l.getMessage().toString().contains("mmchainoptsparse")));
				Assert.assertTrue(heavyHittersContainsSubString(Opcodes.MMCHAIN.toString()) ||
					heavyHittersContainsSubString("sp_mapmmchain"));
			}
			else {
				Assert.assertFalse(log_out.stream().anyMatch(
					l -> l.getMessage().toString().contains("mmchainoptsparse")));
				Assert.assertFalse(heavyHittersContainsSubString(Opcodes.MMCHAIN.toString()) ||
					heavyHittersContainsSubString("sp_mapmmchain"));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = oldFlag1;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = oldFlag2;
			Recompiler.reinitRecompiler();
		}
	}
}
