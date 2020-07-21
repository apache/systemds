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

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class PartialReuseTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "PartialReuse1";
	protected String TEST_CLASS_DIR = TEST_DIR + PartialReuseTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}
	
	@Test
	public void testLineageTrace1CP() {
		//test partial reuse in CP (i.e., w/o reuse-aware recompilation)
		testLineageTraceReuse(TEST_NAME1, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testLineageTrace1Hybrid() {
		//test partial reuse in Hybrid (i.e., w/ reuse-aware recompilation)
		testLineageTraceReuse(TEST_NAME1, ExecMode.HYBRID);
	}

	
	public void testLineageTraceReuse(String testname, ExecMode et) {
		ExecMode execModeOld = setExecMode(et);
		
		try {
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			
			// Without lineage-based reuse enabled
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(output("X"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> X_orig = readDMLMatrixFromHDFS("X");
			
			// With lineage-based reuse enabled
			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(ReuseCacheType.REUSE_HYBRID.name().toLowerCase());
			proArgs.add("-args");
			proArgs.add(output("X"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			Lineage.resetInternalState();
			Lineage.setLinReuseFull();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> X_reused = readDMLMatrixFromHDFS("X");
			Lineage.setLinReuseNone();
			
			//compare matrices
			TestUtils.compareMatrices(X_orig, X_reused, 1e-6, "Origin", "Reused");
			
			//check no evictions (previously buffer pool leak)
			Assert.assertEquals(0, CacheStatistics.getFSWrites());
			//if compiler assisted reuse check for the introduced appends (3x per iteration)
			if( et == ExecMode.HYBRID )
				Assert.assertEquals(900, Statistics.getCPHeavyHitterCount("append"));
		}
		finally {
			resetExecMode(execModeOld);
			Recompiler.reinitRecompiler();
		}
	}
}
