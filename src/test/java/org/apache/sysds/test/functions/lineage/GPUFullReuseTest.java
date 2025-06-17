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

package org.apache.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import jcuda.runtime.cudaError;

public class GPUFullReuseTest extends AutomatedTestBase{
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "LineageReuseGPU";
	protected static final int TEST_VARIANTS = 4;
	protected String TEST_CLASS_DIR = TEST_DIR + GPUFullReuseTest.class.getSimpleName() + "/";

	/*@BeforeClass
	public static void checkGPU() {
		// Skip all the tests if no GPU is available
		// FIXME: Fails to skip if gpu available but no libraries
		Assume.assumeTrue(TestUtils.isGPUAvailable() == cudaError.cudaSuccess);
	}*/

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void ReuseAggBin() {           //reuse AggregateBinary and sum
		testLineageTraceExec(TEST_NAME+"1");
	}

	@Test
	public void ReuseSimpleHLM() {        //hyper-parameter tuning over LM (simple)
		testLineageTraceExec(TEST_NAME+"2");
	}

	@Test
	public void ReuseFunction() {         //multi-level reuse for GPU
		testLineageTraceExec(TEST_NAME+"3");
	}

	@Test
	public void ReuseGridSearchLM() {     //grid search HO for LM
		testLineageTraceExec(TEST_NAME+"4");
	}

	private void testLineageTraceExec(String testname) {
		System.out.println("------------ BEGIN " + testname + "------------");
		getAndLoadTestConfiguration(testname);

		AutomatedTestBase.TEST_GPU = true;  //adds '-gpu'
		List<String> proArgs = new ArrayList<>();
		//proArgs.add("-explain");
		proArgs.add("-stats");
		proArgs.add("-args");
		proArgs.add(output("R"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		fullDMLScriptName = getScript();

		Lineage.resetInternalState();
		//run the test
		OptimizerUtils.ASYNC_PREFETCH = true;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		HashMap<MatrixValue.CellIndex, Double> R_orig = readDMLMatrixFromOutputDir("R");

		proArgs.clear();
		proArgs.add("-stats");
		proArgs.add("-lineage");
		proArgs.add(LineageCacheConfig.ReuseCacheType.REUSE_MULTILEVEL.name().toLowerCase());
		proArgs.add("-args");
		proArgs.add(output("R"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		fullDMLScriptName = getScript();

		Lineage.resetInternalState();
		//run the test
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		OptimizerUtils.ASYNC_PREFETCH = false;
		AutomatedTestBase.TEST_GPU = false;
		HashMap<MatrixValue.CellIndex, Double> R_reused = readDMLMatrixFromOutputDir("R");

		//compare results
		TestUtils.compareMatrices(R_orig, R_reused, 1e-6, "Origin", "Reused");

		if( testname.endsWith("3") ) { //function reuse
			Assert.assertEquals(1L, LineageCacheStatistics.getMultiLevelFnHits());
		}
	}
}

