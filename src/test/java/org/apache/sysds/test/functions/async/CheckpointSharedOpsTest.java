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

	import org.apache.sysds.common.Types;
	import org.apache.sysds.hops.OptimizerUtils;
	import org.apache.sysds.hops.recompile.Recompiler;
	import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
	import org.apache.sysds.runtime.matrix.data.MatrixValue;
	import org.apache.sysds.test.AutomatedTestBase;
	import org.apache.sysds.test.TestConfiguration;
	import org.apache.sysds.test.TestUtils;
	import org.apache.sysds.utils.Statistics;
	import org.junit.Assert;
	import org.junit.Test;

public class CheckpointSharedOpsTest extends AutomatedTestBase {

	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "CheckpointSharedOps";
	protected static final int TEST_VARIANTS = 2;
	protected static String TEST_CLASS_DIR = TEST_DIR + CheckpointSharedOpsTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=TEST_VARIANTS; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void test1() {
		// Shared cpmm/rmm between two jobs
		runTest(TEST_NAME+"1");
	}

	@Test
	public void testPnmf() {
		// Place checkpoint at the end of a loop as the updated vars are read in each iteration.
		runTest(TEST_NAME+"2");
	}

	public void runTest(String testname) {
		Types.ExecMode oldPlatform = setExecMode(Types.ExecMode.HYBRID);

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
			long numCP = Statistics.getCPHeavyHitterCount("sp_chkpoint");

			OptimizerUtils.ASYNC_CHECKPOINT_SPARK = true;
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_mp = readDMLScalarFromOutputDir("R");
			long numCP_maxp = Statistics.getCPHeavyHitterCount("sp_chkpoint");
			OptimizerUtils.ASYNC_CHECKPOINT_SPARK = false;

			//compare matrices
			boolean matchVal = TestUtils.compareMatrices(R, R_mp, 1e-3, "Origin", "withChkpoint");
			if (!matchVal)
				System.out.println("Value w/o Checkpoint "+R+" w/ Checkpoint "+R_mp);
			//compare checkpoint instruction count
			if (!testname.equalsIgnoreCase(TEST_NAME+"2"))
				Assert.assertTrue("Violated checkpoint count: " + numCP + " < " + numCP_maxp, numCP < numCP_maxp);
		} finally {
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
			Recompiler.reinitRecompiler();
		}
	}
}
