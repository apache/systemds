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


package org.apache.sysds.test.functions.caching;

	import java.util.ArrayList;
	import java.util.HashMap;
	import java.util.List;

	import org.apache.sysds.hops.OptimizerUtils;
	import org.apache.sysds.hops.recompile.Recompiler;
	import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
	import org.apache.sysds.runtime.controlprogram.caching.UnifiedMemoryManager;
	import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
	import org.apache.sysds.runtime.matrix.data.MatrixValue;
	import org.apache.sysds.test.AutomatedTestBase;
	import org.apache.sysds.test.TestConfiguration;
	import org.apache.sysds.test.TestUtils;
	import org.junit.Assert;
	import org.junit.Test;

public class UMMTest extends AutomatedTestBase {

	protected static final String TEST_DIR = "functions/caching/";
	protected static final String TEST_NAME1 = "UMMTest1";

	protected String TEST_CLASS_DIR = TEST_DIR + org.apache.sysds.test.functions.caching.UMMTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}

	@Test
	//@Ignore
	public void testEvictionOrder() {
		runTest(TEST_NAME1);
	}

	public void runTest(String testname) {

		try {
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			long oldBufferPool = (long)(0.15 * InfrastructureAnalyzer.getLocalMaxMemory());

			// Static memory management
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(String.valueOf(oldBufferPool));
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			//HashMap<MatrixValue.CellIndex, Double> R_static = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> R_static = readDMLScalarFromOutputDir("R");
			long FSwrites_static = CacheStatistics.getFSWrites();

			// Unified memory management (cache size = 85% of heap)
			//UnifiedMemoryManager.setUMMLimit((long)(0.85 * InfrastructureAnalyzer.getLocalMaxMemory()));
			//CacheableData.UMM = true;
			//UnifiedMemoryManager.cleanup();
			//LazyWriteBuffer.cleanup();
			OptimizerUtils.enableUMM();
			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-args");
			proArgs.add(String.valueOf(oldBufferPool));
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			UnifiedMemoryManager.cleanup();
			//HashMap<MatrixValue.CellIndex, Double> R_unified = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> R_unified= readDMLScalarFromOutputDir("R");
			long FSwrites_unified = CacheStatistics.getFSWrites();

			// Compare results
			TestUtils.compareMatrices(R_static, R_unified, 1e-6, "static", "unified");
			// Compare FS write counts (#unified FS writes always smaller than #static FS writes)
			Assert.assertTrue("Violated buffer pool eviction counts: "+FSwrites_unified+" <= "+FSwrites_static,
				FSwrites_unified <= FSwrites_static);
		}
		finally {
			Recompiler.reinitRecompiler();
		}
	}
}

