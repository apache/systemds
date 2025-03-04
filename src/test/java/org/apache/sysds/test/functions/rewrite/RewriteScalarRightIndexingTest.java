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

package org.apache.sysds.test.functions.rewrite;


import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;

public class RewriteScalarRightIndexingTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/rewrite/";
	private final static String TEST_NAME = "RewriteScalarRightIndexing";
	
	private final static String TEST_CLASS_DIR = TEST_DIR + RewriteScalarRightIndexingTest.class.getSimpleName() + "/";
	
	private final static int rows = 122;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A"}));
	}

	@Test
	public void testScalarRightIndexingCP() {
		runScalarRightIndexing(true, ExecType.CP);
	}
	
	@Test
	public void testScalarRightIndexingNoRewriteCP() {
		runScalarRightIndexing(false, ExecType.CP);
	}
	
	@Test
	public void testScalarRightIndexingSpark() {
		runScalarRightIndexing(true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarRightIndexingNoRewriteSpark() {
		runScalarRightIndexing(false, ExecType.SPARK);
	}
	
	private void runScalarRightIndexing(boolean rewrite, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		boolean flagOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrite;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args",
				Long.toString(rows), output("A")};
			runTest(true, false, null, -1);
			
			Double ret = readDMLScalarFromOutputDir("A").get(new CellIndex(1,1));
			Assert.assertEquals(Double.valueOf(103.0383), ret, 1e-4);
			if(rewrite) //w/o rewrite 122 casts
				Assert.assertTrue(Statistics.getCPHeavyHitterCount("castdts")<=1);
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = flagOld;
		}
	}
}
