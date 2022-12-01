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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class MaxParallelizeOrderTest extends AutomatedTestBase {

	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "MaxParallelizeOrder";
	protected static final int TEST_VARIANTS = 4;
	protected static String TEST_CLASS_DIR = TEST_DIR + MaxParallelizeOrderTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=TEST_VARIANTS; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void testlmds() {
		runTest(TEST_NAME+"1");
	}

	@Test
	public void testl2svm() {
		runTest(TEST_NAME+"2");
	}

	@Test
	public void testSparkAction() {
		runTest(TEST_NAME+"3");
	}

	@Test
	public void testSparkTransformations() {
		runTest(TEST_NAME+"4");
	}

	public void runTest(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		boolean old_trans_exec_type = OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE;
		ExecMode oldPlatform = setExecMode(ExecMode.HYBRID);

		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		long mem = 1024*1024*8;
		InfrastructureAnalyzer.setLocalMaxMemory(mem);

		try {
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			List<String> proArgs = new ArrayList<>();

			proArgs.add("-explain");
			//proArgs.add("recompile_runtime");
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");

			OptimizerUtils.ASYNC_PREFETCH_SPARK = true;
			OptimizerUtils.MAX_PARALLELIZE_ORDER = true;
			if (testname.equalsIgnoreCase(TEST_NAME+"4"))
				OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = false;
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_mp = readDMLScalarFromOutputDir("R");
			OptimizerUtils.ASYNC_PREFETCH_SPARK = false;
			OptimizerUtils.MAX_PARALLELIZE_ORDER = false;
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = true;

			//compare matrices
			boolean matchVal = TestUtils.compareMatrices(R, R_mp, 1e-6, "Origin", "withPrefetch");
			if (!matchVal)
				System.out.println("Value w/o Prefetch "+R+" w/ Prefetch "+R_mp);
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = old_trans_exec_type;
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
			Recompiler.reinitRecompiler();
		}
	}
}
