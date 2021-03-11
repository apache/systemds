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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import jcuda.runtime.cudaError;

public class LineageTraceGPUTest extends AutomatedTestBase{
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "LineageTraceGPU1"; 
	protected String TEST_CLASS_DIR = TEST_DIR + LineageTraceGPUTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 5;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
	}
	
	@Test
	public void simpleHLM_gpu() {              //hyper-parameter tuning over LM (simple)
		testLineageTraceExec(TEST_NAME1);
	}
	
	private void testLineageTraceExec(String testname) {
		System.out.println("------------ BEGIN " + testname + "------------");
		
		int gpuStatus = TestUtils.isGPUAvailable(); 
		getAndLoadTestConfiguration(testname);
		List<String> proArgs = new ArrayList<>();
		
		proArgs.add("-stats");
		if (gpuStatus == cudaError.cudaSuccess)
			proArgs.add("-gpu");
		proArgs.add("-lineage");
		proArgs.add("-args");
		proArgs.add(output("R"));
		proArgs.add(String.valueOf(numRecords));
		proArgs.add(String.valueOf(numFeatures));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		fullDMLScriptName = getScript();
		
		Lineage.resetInternalState();
		//run the test
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		
		//get lineage and generate program
		String Rtrace = readDMLLineageFromHDFS("R");
		//NOTE: the generated program is CP-only.
		Data ret = LineageRecomputeUtils.parseNComputeLineageTrace(Rtrace, null);
		
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
		MatrixBlock tmp = ((MatrixObject)ret).acquireReadAndRelease();
		TestUtils.compareMatrices(dmlfile, tmp, 1e-6);
	}
}
