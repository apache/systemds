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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageParser;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Explain;

import java.util.ArrayList;
import java.util.List;

public class LineageTraceSparkTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "LineageTraceSpark1";
	protected static final String TEST_NAME2 = "LineageTraceSpark2";
	protected static final String TEST_NAME3 = "LineageTraceSpark3";
	protected static final String TEST_NAME4 = "LineageTraceSpark4";
	protected static final String TEST_NAME5 = "LineageTraceSpark5";
	protected static String TEST_CLASS_DIR = TEST_DIR + LineageTraceSparkTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 5;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"X", "X.lineage", "Y", "Y.lineage"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"X", "X.lineage", "Y", "Y.lineage"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"X", "X.lineage", "Y", "Y.lineage"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[]{"X", "X.lineage", "Y", "Y.lineage"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[]{"X", "X.lineage", "Y", "Y.lineage"}));
	}
	
	@Test
	public void testLineageTraceSpark1() {
		testLineageTraceSpark(TEST_NAME1);
	}
	
	@Test
	public void testLineageTraceSpark2() {
		testLineageTraceSpark(TEST_NAME2);
	}
	
	@Test
	public void testLineageTraceSpark3() {
		testLineageTraceSpark(TEST_NAME3);
	}
	
	@Test
	public void testLineageTraceSpark4() {
		testLineageTraceSpark(TEST_NAME4);
	}
	
	@Test
	public void testLineageTraceSpark5() {
		testLineageTraceSpark(TEST_NAME5);
	}
	
	
	public void testLineageTraceSpark(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		Types.ExecMode oldPlatform = rtplatform;
		boolean oldLocalSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			rtplatform = Types.ExecMode.SPARK;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			int rows = numRecords;
			int cols = numFeatures;
			
			getAndLoadTestConfiguration(testname);
			
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(output("X"));
			proArgs.add(output("Y"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			
			fullDMLScriptName = getScript();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			String X_lineage = readDMLLineageFromHDFS("X");
			String Y_lineage = readDMLLineageFromHDFS("Y");
			
			LineageItem X_li = LineageParser.parseLineageTrace(X_lineage);
			LineageItem Y_li = LineageParser.parseLineageTrace(Y_lineage);
			
			TestUtils.compareScalars(X_lineage, Explain.explain(X_li));
			TestUtils.compareScalars(Y_lineage, Explain.explain(Y_li));
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			rtplatform = oldPlatform;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldLocalSparkConfig;
			Recompiler.reinitRecompiler();
		}
	}
}
