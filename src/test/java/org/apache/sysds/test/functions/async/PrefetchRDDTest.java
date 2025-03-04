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
import org.junit.Assert;
import org.junit.Test;

public class PrefetchRDDTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "PrefetchRDD";
	protected static final int TEST_VARIANTS = 5;
	protected static String TEST_CLASS_DIR = TEST_DIR + PrefetchRDDTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}
	
	@Test
	public void testAsyncSparkOPs1() {
		//Single CP consumer. Prefetch Lop has one output.
		runTest(TEST_NAME+"1");
	}

	@Test
	public void testAsyncSparkOPs2() {
		//Two CP consumers. Prefetch Lop has two outputs.
		runTest(TEST_NAME+"2");
	}

	@Test
	public void testAsyncSparkOPs3() {
		//SP binary consumer, followed by an action. No Prefetch.
		runTest(TEST_NAME+"3");
	}

	@Test
	public void testAsyncSparkOPs4() {
		//SP consumer. Collect to broadcast to the SP consumer. Prefetch.
		runTest(TEST_NAME+"4");
	}

	@Test
	public void testAsyncSparkOPs5() {
		//List type consumer. No Prefetch.
		runTest(TEST_NAME+"5");
	}

	public void runTest(String testname) {
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
			
			proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");

			OptimizerUtils.ASYNC_PREFETCH = true;
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			OptimizerUtils.ASYNC_PREFETCH = false;
			OptimizerUtils.MAX_PARALLELIZE_ORDER = false;
			HashMap<MatrixValue.CellIndex, Double> R_pf = readDMLScalarFromOutputDir("R");

			//compare matrices
			boolean matchVal = TestUtils.compareMatrices(R, R_pf, 1e-6, "Origin", "withPrefetch");
			if (!matchVal)
				System.out.println("Value w/o Prefetch "+R+" w/ Prefetch "+R_pf);
			//assert Prefetch instructions and number of success.
			long expected_numPF = 1;
			if (testname.equalsIgnoreCase(TEST_NAME+"3")
				|| testname.equalsIgnoreCase(TEST_NAME+"5"))
				expected_numPF = 0;
			//long expected_successPF = !testname.equalsIgnoreCase(TEST_NAME+"3") ? 1 : 0;
			long numPF = Statistics.getCPHeavyHitterCount(Opcodes.PREFETCH.toString());
			Assert.assertTrue("Violated Prefetch instruction count: "+numPF, numPF == expected_numPF);
			//long successPF = SparkStatistics.getAsyncPrefetchCount();
			//Assert.assertTrue("Violated successful Prefetch count: "+successPF, successPF == expected_successPF);
		} finally {
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = old_trans_exec_type;
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
			Recompiler.reinitRecompiler();
		}
	}
}