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
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DedupReuseTest extends AutomatedTestBase
{
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "DedupReuse"; 
	protected static final String TEST_NAME1 = "DedupReuse1"; 
	protected static final String TEST_NAME2 = "DedupReuse2"; 
	protected static final String TEST_NAME3 = "DedupReuse3"; 
	protected static final String TEST_NAME4 = "DedupReuse4"; 
	protected static final String TEST_NAME5 = "DedupReuse5"; 
	protected static final String TEST_NAME6 = "DedupReuse6"; 

	protected String TEST_CLASS_DIR = TEST_DIR + DedupReuseTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 100;
	protected static final int numFeatures = 50;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=6; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}
	
	@Test
	public void testLineageTrace1() {
		// Reuse operations from outside to a dedup-loop
		testLineageTrace(TEST_NAME1);
	}

	@Test
	public void testLineageTrace5() {
		// Reuse operations from outside to a dedup-loop
		testLineageTrace(TEST_NAME5);
	}

	@Test
	public void testLineageTrace2() {
		// Reuse all operations from a dedup loop to another dedup loop
		testLineageTrace(TEST_NAME2);
	}
	
	@Test
	public void testLineageTrace3() {
		// Reuse all operations from a non-dedup-loop to a dedup loop
		testLineageTrace(TEST_NAME3);
	}

	@Test
	public void testLineageTrace4() {
		// Reuse an operation for each iteration of a dedup loop
		testLineageTrace(TEST_NAME4);
	}

	@Test
	public void testLineageTrace6() {
		// Reuse minibatch-wise preprocessing in a mini-batch like scenario
		testLineageTrace(TEST_NAME6);
	}
	
	public void testLineageTrace(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		Types.ExecMode old_rtplatform = AutomatedTestBase.rtplatform;
		
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			AutomatedTestBase.rtplatform = Types.ExecMode.SINGLE_NODE;
			
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			List<String> proArgs;

			//w/o lineage deduplication
			proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("reuse_full");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<CellIndex, Double> orig = readDMLMatrixFromOutputDir("R");
			long hitCount_nd = LineageCacheStatistics.getInstHits();

			//w/ lineage deduplication
			proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("reuse_full");
			proArgs.add("dedup");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);

			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<CellIndex, Double> dedup = readDMLMatrixFromOutputDir("R");
			long hitCount_d = LineageCacheStatistics.getInstHits();

			//match the results
			TestUtils.compareMatrices(orig, dedup, 1e-6, "Original", "Dedup");
			//compare cache hits
			Assert.assertTrue(hitCount_nd >= hitCount_d); //FIXME
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			AutomatedTestBase.rtplatform = old_rtplatform;
			Recompiler.reinitRecompiler(); 
		}
	}
}
