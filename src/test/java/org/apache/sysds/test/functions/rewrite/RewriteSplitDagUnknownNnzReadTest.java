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
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class RewriteSplitDagUnknownNnzReadTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSplitDagUnknownNnz";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSplitDagUnknownNnzReadTest.class.getSimpleName() + "/";
	private static final String PACKAGE_REWRITE = "org.apache.sysds.hops.rewrite.HopRewriteRule";
	private static final String PACKAGE_RECOMPILE = "org.apache.sysds.hops.rewrite.StatementBlockRewriteRule";
	private static Level _oldLevelRewrite;
	private static Level _oldLevelRecompile;

	@Parameterized.Parameter(0)
	public int rows;

	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public double[] sparsities;

	@Parameterized.Parameter(3)
	public double eps;

	@Parameterized.Parameter(4)
	public boolean tsmm;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {rows, cols, sparsities, eps},
			{1000, 300, new double[] {0.10d, 0.10d}, Math.pow(10, -10), false},
			{5, 3, new double[] {0.1, 1}, Math.pow(10, -10), true},});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
		_oldLevelRewrite = Logger.getLogger(PACKAGE_REWRITE).getLevel();
		Logger.getLogger(PACKAGE_REWRITE).setLevel(Level.TRACE);
		_oldLevelRecompile = Logger.getLogger(PACKAGE_RECOMPILE).getLevel();
		Logger.getLogger(PACKAGE_RECOMPILE).setLevel(Level.TRACE);
	}

	@Override
	public void tearDown() {
		super.tearDown();
		Logger.getLogger(PACKAGE_REWRITE).setLevel(_oldLevelRewrite);
		Logger.getLogger(PACKAGE_RECOMPILE).setLevel(_oldLevelRecompile);
	}

	@Test
	public void testSplitDagUnknownNnzRead() {
		boolean oldFlag1 = OptimizerUtils.ALLOW_TRANSPOSE_MMCHAIN_REWRITES;
		boolean oldFlag2 = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		DMLConfig oldDMLConfig = ConfigurationManager.getDMLConfig();

		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "hops", "-stats",
				"-args", input("X"), input("Y"), output("R")};

			OptimizerUtils.ALLOW_TRANSPOSE_MMCHAIN_REWRITES = true;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = true;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsities[0], 7);
			double[][] Y = getRandomMatrix(cols, 1, -1, 1, sparsities[1], 3);
			long Y_nnz = Stream.of(Y).mapToLong(row -> DoubleStream.of(row).filter(val -> val != 0).count()).sum();
			writeInputMatrixWithMTD("X", X, true); // write matrix characteristics w/ missing nnz
			writeInputMatrixWithMTD("Y", Y, Y_nnz, true);

			// run reference test
			TestAppender appender = LoggingUtils.overwrite(); // capture log output
			runTest(true, false, null, -1);
			List<LoggingEvent> log_reference = LoggingUtils.reinsert(appender); // revert the logger to print to stdout

			// get reference result matrix
			HashMap<MatrixValue.CellIndex, Double> referenceResult = readDMLMatrixFromOutputDir("R");

			try {
				DMLConfig dmlConfig = new DMLConfig(getCurConfigFile().getPath());
				dmlConfig.setTextValue(DMLConfig.SPARSITY_REWRITES, "true");
				dmlConfig.setTextValue(DMLConfig.SPARSITY_RECOMPILE, "true");
				overwriteCurrentConfig(dmlConfig);
			}
			catch(FileNotFoundException fnfe) {
				Assert.fail("Could not find DML config file: " +
					getCurConfigFile().getPath() + " . " + fnfe.getMessage());
			}
			catch(IOException ioe) {
				Assert.fail("Could not overwrite the DML configuration file. " + ioe.getMessage());
			}

			// run original test
			appender = LoggingUtils.overwrite(); // capture log output
			runTest(true, false, null, -1);
			List<LoggingEvent> log_original = LoggingUtils.reinsert(appender); // revert the logger to print to stdout

			// get resulting matrix
			HashMap<MatrixValue.CellIndex, Double> originalResult = readDMLMatrixFromOutputDir("R");

			// compare matrices
			TestUtils.compareMatrices(originalResult, referenceResult, eps, "Stat-Original", "Stat-Reference");

			final String DELIMITER = "\n";
			// check original log output
			String log_original_string = String.join(DELIMITER,
				log_original.stream().map(l -> l.getMessage().toString()).toArray(String[]::new));
			Assert.assertTrue(log_original_string.contains("Applied splitDagUnknownNnzRead."));
			Assert.assertTrue(log_original_string.contains("mmchainoptsparse"));
			Assert.assertTrue(log_original_string.contains("Optimal Sparse MM Chain:"));
			// check reference log output
			String log_reference_string = String.join(DELIMITER,
				log_reference.stream().map(l -> l.getMessage().toString()).toArray(String[]::new));
			Assert.assertFalse(log_reference_string.contains("splitDagUnknownNnzRead"));
			Assert.assertFalse(log_reference_string.contains("mmchainoptsparse"));
		}
		finally {
			OptimizerUtils.ALLOW_TRANSPOSE_MMCHAIN_REWRITES = oldFlag1;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = oldFlag2;
			try {
				overwriteCurrentConfig(oldDMLConfig);
			}
			catch(IOException ioe) {
				Assert.fail("Unable to restore the previous DML configuration. " + ioe.getMessage());
			}
			Recompiler.reinitRecompiler();
		}
	}

	private void overwriteCurrentConfig(DMLConfig config) throws IOException {
		Files.write(getCurConfigFile().toPath(), config.serializeDMLConfig().getBytes(StandardCharsets.UTF_8));
	}
}
