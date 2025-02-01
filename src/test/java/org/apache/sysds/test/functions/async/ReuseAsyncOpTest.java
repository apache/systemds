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

package org.apache.sysds.test.functions.async;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assert;
import org.junit.Test;

public class ReuseAsyncOpTest extends AutomatedTestBase {
	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "ReuseAsyncOp";
	protected static final int TEST_VARIANTS = 2;
	protected static String TEST_CLASS_DIR = TEST_DIR + ReuseAsyncOpTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=TEST_VARIANTS; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void testReusePrefetch() {
		// Reuse prefetch results
		runTest(TEST_NAME+"1", ExecMode.HYBRID, 1);
	}

	@Test
	public void testlmds() {
		// Reuse future-based tsmm and mapmm
		runTest(TEST_NAME+"2", ExecMode.HYBRID, 2);
	}

	public void runTest(String testname, ExecMode execMode, int testId) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		boolean old_trans_exec_type = OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE;
		ExecMode oldPlatform = setExecMode(ExecMode.HYBRID);
		rtplatform = execMode;

		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		long mem = 1024*1024*8;
		InfrastructureAnalyzer.setLocalMaxMemory(mem);

		try {
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			List<String> proArgs = new ArrayList<>();

			proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			Lineage.resetInternalState();
			enableAsync(); //enable max_reuse and prefetch
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			disableAsync();
			HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");
			long numTsmm = Statistics.getCPHeavyHitterCount("sp_tsmm");
			long numMapmm = Statistics.getCPHeavyHitterCount("sp_mapmm");
			long numPrefetch = Statistics.getCPHeavyHitterCount(Opcodes.PREFETCH.toString());

			proArgs.clear();
			proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(LineageCacheConfig.ReuseCacheType.REUSE_FULL.name().toLowerCase());
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			Lineage.resetInternalState();
			enableAsync(); //enable max_reuse and prefetch
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			disableAsync();
			HashMap<MatrixValue.CellIndex, Double> R_reused = readDMLScalarFromOutputDir("R");
			long numTsmm_r = Statistics.getCPHeavyHitterCount("sp_tsmm");
			long numMapmm_r = Statistics.getCPHeavyHitterCount("sp_mapmm");
			long numPrefetch_r = Statistics.getCPHeavyHitterCount(Opcodes.PREFETCH.toString());

			//compare matrices
			boolean matchVal = TestUtils.compareMatrices(R, R_reused, 1e-6, "Origin", "withPrefetch");
			if (!matchVal)
				System.out.println("Value w/o reuse "+R+" w/ reuse "+R_reused);
			if (testId == 2) {
				Assert.assertTrue("Violated sp_tsmm reuse count: " + numTsmm_r + " < " + numTsmm, numTsmm_r < numTsmm);
				Assert.assertTrue("Violated sp_mapmm reuse count: " + numMapmm_r + " < " + numMapmm, numMapmm_r < numMapmm);
			}
			if (testId == 1)
				Assert.assertTrue("Violated prefetch reuse count: " + numPrefetch_r + " < " + numPrefetch, numPrefetch_r<numPrefetch);

		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = old_trans_exec_type;
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
			Recompiler.reinitRecompiler();
		}
	}

	private void enableAsync() {
		OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = false;
		OptimizerUtils.MAX_PARALLELIZE_ORDER = true;
		OptimizerUtils.ASYNC_PREFETCH = true;
	}

	private void disableAsync() {
		OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = true;
		OptimizerUtils.MAX_PARALLELIZE_ORDER = false;
		OptimizerUtils.ASYNC_PREFETCH = false;
	}
}
