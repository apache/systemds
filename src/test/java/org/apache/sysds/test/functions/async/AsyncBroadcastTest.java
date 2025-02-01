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
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.stats.SparkStatistics;
import org.junit.Assert;
import org.junit.Test;

public class AsyncBroadcastTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "BroadcastVar";
	protected static final int TEST_VARIANTS = 2;
	protected static String TEST_CLASS_DIR = TEST_DIR + AsyncBroadcastTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}
	
	@Test
	public void testAsyncBroadcast1() {
		runTest(TEST_NAME+"1");
	}

	@Test
	public void testAsyncBroadcast2() {
		runTest(TEST_NAME+"2");
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
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = false;
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			
			List<String> proArgs = new ArrayList<>();
			
			//proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");

			OptimizerUtils.ASYNC_BROADCAST_SPARK = true;
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			OptimizerUtils.ASYNC_BROADCAST_SPARK = false;
			HashMap<MatrixValue.CellIndex, Double> R_bc = readDMLScalarFromOutputDir("R");

			//compare matrices
			TestUtils.compareMatrices(R, R_bc, 1e-6, "Origin", "withBroadcast");

			//assert called and successful early broadcast counts
			long expected_numBC = 1;
			long expected_successBC = 1;
			long numBC = Statistics.getCPHeavyHitterCount(Opcodes.BROADCAST.toString());
			Assert.assertTrue("Violated Broadcast instruction count: "+numBC, numBC == expected_numBC);
			long successBC = SparkStatistics.getAsyncBroadcastCount();
			Assert.assertTrue("Violated successful Broadcast count: "+successBC, successBC == expected_successBC);
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
