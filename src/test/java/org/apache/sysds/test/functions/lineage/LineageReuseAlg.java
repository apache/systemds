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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LineageReuseAlg extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "LineageReuseAlg";
	protected static final int TEST_VARIANTS = 6;
	protected String TEST_CLASS_DIR = TEST_DIR + LineageReuseAlg.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=TEST_VARIANTS; i++ )
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void testStepLMHybrid() {
		testLineageTrace(TEST_NAME+"1", ReuseCacheType.REUSE_HYBRID);
	}
	
	@Test
	public void testGridSearchLMHybrid() {
		testLineageTrace(TEST_NAME+"2", ReuseCacheType.REUSE_HYBRID);
	}

	@Test
	public void testMultiLogRegHybrid() {
		testLineageTrace(TEST_NAME+"3", ReuseCacheType.REUSE_HYBRID);
	}

	@Test
	public void testPCAHybrid() {
		testLineageTrace(TEST_NAME+"4", ReuseCacheType.REUSE_HYBRID);
	}

	@Test
	public void testGridSearchL2svmHybrid() {
		testLineageTrace(TEST_NAME+"5", ReuseCacheType.REUSE_HYBRID);
	}

	@Test
	public void testPCA_LM_pipeline() {
		testLineageTrace(TEST_NAME+"6", ReuseCacheType.REUSE_HYBRID);
	}
	
	@Test
	public void testStepLMFull() {
		testLineageTrace(TEST_NAME+"1", ReuseCacheType.REUSE_FULL);
	}
	
	@Test
	public void testGridSearchLMFull() {
		testLineageTrace(TEST_NAME+"2", ReuseCacheType.REUSE_FULL);
	}

	@Test
	public void testMultiLogRegFull() {
		testLineageTrace(TEST_NAME+"3", ReuseCacheType.REUSE_FULL);
	}

	@Test
	public void testPCAFull() {
		testLineageTrace(TEST_NAME+"4", ReuseCacheType.REUSE_FULL);
	}

	public void testLineageTrace(String testname, ReuseCacheType reuseType) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		ExecMode platformOld = setExecMode(ExecType.CP);
		
		try {
			System.out.println("------------ BEGIN " + testname + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			
			// Without lineage-based reuse enabled
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-stats");
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
			proArgs.add(reuseType.name().toLowerCase());
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
			rtplatform = platformOld;
			Recompiler.reinitRecompiler();
		}
	}
}
