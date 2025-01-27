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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class CacheEvictionTest extends LineageBase {

	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "CacheEviction2";

	protected String TEST_CLASS_DIR = TEST_DIR + CacheEvictionTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-eviction.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}
	
	@Test
	public void testEvictionOrder() {
		runTest(TEST_NAME1);
	}

	public void runTest(String testname) {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			LOG.debug("------------ BEGIN " + testname + "------------");
			
			/* This test verifies the order of evicted items w.r.t. the specified
			 * cache policies, using a mini-batch wise autoencoder inspired
			 * test script. An epoch-wise reusable scale and shift is part of
			 * every batch processing. LRU fails to reuse the scale calls as
			 * it tends to evicts scale and shift intermediates due to higher
			 * number of post scale intermediates, where cost & size successfully
			 * reuses all the reusable operations.
			 * 
			 * TODO: add DagHeight. All three policies perform as expected in my
			 * laptop, but for some reasons, LRU performs better in github actions
			 * - that leads to failed comparison between dagheight and LRU.
			 */
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			Lineage.resetInternalState();
			
			// LRU based eviction
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(ReuseCacheType.REUSE_FULL.name().toLowerCase());
			proArgs.add("policy_lru");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_lru = readDMLMatrixFromOutputDir("R");
			long hitCount_lru = LineageCacheStatistics.getInstHits();
			long colmeanCount_lru = Statistics.getCPHeavyHitterCount(Opcodes.UACMEAN.toString());
			
			// costnsize scheme (computationTime/Size)
			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(ReuseCacheType.REUSE_FULL.name().toLowerCase());
			proArgs.add("policy_costnsize");
			proArgs.add("-args");
			proArgs.add(output("R"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			Lineage.resetInternalState();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_costnsize= readDMLMatrixFromOutputDir("R");
			long hitCount_cs = LineageCacheStatistics.getInstHits();
			long colmeanCount_cs = Statistics.getCPHeavyHitterCount(Opcodes.UACMEAN.toString());
			
			// Compare results
			Lineage.setLinReuseNone();
			TestUtils.compareMatrices(R_lru, R_costnsize, 1e-6, "LRU", "costnsize");
			// Compare cache hits
			Assert.assertTrue("Violated cache hit count: "+hitCount_lru+" < "+hitCount_cs, 
					hitCount_lru < hitCount_cs);
			// Compare reused instruction (uacmean) counts
			Assert.assertTrue("Violated uacmean count: "+colmeanCount_cs+" < "+colmeanCount_lru, 
					colmeanCount_cs < colmeanCount_lru);
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
			Recompiler.reinitRecompiler();
		}
	}
	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
