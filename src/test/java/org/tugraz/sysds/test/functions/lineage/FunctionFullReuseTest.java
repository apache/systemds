/*
 * Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.lineage;

import org.junit.Test;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.runtime.lineage.Lineage;
import org.tugraz.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class FunctionFullReuseTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "FunctionFullReuse1";
	protected static final String TEST_NAME2 = "FunctionFullReuse2";
	protected static final String TEST_NAME3 = "FunctionFullReuse3";
	protected static final String TEST_NAME4 = "FunctionFullReuse4";
	protected static final String TEST_NAME5 = "FunctionFullReuse5";
	protected String TEST_CLASS_DIR = TEST_DIR + FunctionFullReuseTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5));
	}
	
	@Test
	public void testCacheHit() {
		testLineageTrace(TEST_NAME1);
	}
	
	@Test
	public void testCacheMiss() {
		testLineageTrace(TEST_NAME2);
	}

	@Test
	public void testMultipleReturns() {
		testLineageTrace(TEST_NAME3);
	}

	@Test
	public void testNestedFunc() {
		testLineageTrace(TEST_NAME4);
	}

	/*@Test
	public void testStepLM() {
		testLineageTrace(TEST_NAME5);
	}*/
	
	public void testLineageTrace(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
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
			
			TestUtils.compareMatrices(X_orig, X_reused, 1e-6, "Origin", "Reused");
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			Recompiler.reinitRecompiler();
		}
	}
}

