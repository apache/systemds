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

package org.apache.sysds.test.functions.compress.workload;

import static org.junit.Assert.fail;

import java.io.File;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.compress.workload.WorkloadAnalyzer;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class WorkloadAlgorithmTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(WorkloadAnalysisTest.class.getName());

	private final static String TEST_NAME1 = "WorkloadAnalysisMLogReg";
	private final static String TEST_NAME2 = "WorkloadAnalysisLmDS";
	private final static String TEST_NAME3 = "WorkloadAnalysisPCA";
	private final static String TEST_NAME4 = "WorkloadAnalysisSliceLine";
	private final static String TEST_NAME5 = "WorkloadAnalysisSliceFinder";
	private final static String TEST_NAME6 = "WorkloadAnalysisLmCG";
	private final static String TEST_DIR = "functions/compress/workload/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WorkloadAnalysisTest.class.getSimpleName() + "/";

	private int nRows = 1000;

	private double[][] X;
	private double[][] y;

	public WorkloadAlgorithmTest() {
		X = TestUtils.round(getRandomMatrix(nRows, 20, 0, 5, 1.0, 7));
		y = getRandomMatrix(nRows, 1, 0, 0, 1.0, 3);

		for(int i = 0; i < X.length; i++)
			y[i][0] = Math.max(X[i][0], 1);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"B"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"B"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"B"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"B"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {"B"}));
	}

	@Test
	public void testMLogRegCP() {
		runWorkloadAnalysisTest(TEST_NAME1, ExecMode.HYBRID, 2, false);
	}

	@Test
	public void testLmSP() {
		runWorkloadAnalysisTest(TEST_NAME2, ExecMode.SPARK, 2, false);
	}

	@Test
	public void testLmCP() {
		runWorkloadAnalysisTest(TEST_NAME2, ExecMode.HYBRID, 2, false);
	}

	@Test
	public void testLmDSSP() {
		runWorkloadAnalysisTest(TEST_NAME2, ExecMode.SPARK, 2, false);
	}

	@Test
	public void testLmDSCP() {
		runWorkloadAnalysisTest(TEST_NAME2, ExecMode.HYBRID, 2, false);
	}

	@Test
	public void testPCASP() {
		runWorkloadAnalysisTest(TEST_NAME3, ExecMode.SPARK, 1, false);
	}

	@Test
	public void testPCACP() {
		runWorkloadAnalysisTest(TEST_NAME3, ExecMode.HYBRID, 1, false);
	}

	@Test
	public void testSliceLineCP1() {
		runWorkloadAnalysisTest(TEST_NAME4, ExecMode.HYBRID, 0, false);
	}

	@Test
	public void testSliceLineCP2() {
		runWorkloadAnalysisTest(TEST_NAME4, ExecMode.HYBRID, 2, true);
	}

	@Test
	public void testLmCGSP() {
		runWorkloadAnalysisTest(TEST_NAME6, ExecMode.SPARK, 2, false);
	}
	
	@Test
	public void testLmCGCP() {
		runWorkloadAnalysisTest(TEST_NAME6, ExecMode.HYBRID, 2, false);
	}
	
	// private void runWorkloadAnalysisTest(String testname, ExecMode mode, int compressionCount) {
	private void runWorkloadAnalysisTest(String testname, ExecMode mode, int compressionCount, boolean intermediates) {
		ExecMode oldPlatform = setExecMode(mode);
		boolean oldIntermediates = WorkloadAnalyzer.ALLOW_INTERMEDIATE_CANDIDATES;
		
		try {
			loadTestConfiguration(getTestConfiguration(testname));
			WorkloadAnalyzer.ALLOW_INTERMEDIATE_CANDIDATES = intermediates;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-stats", "20", "-args", input("X"), input("y"), output("B")};

			writeInputMatrixWithMTD("X", X, false);
			writeInputMatrixWithMTD("y", y, false);

			String ret = runTest(null).toString();
			LOG.debug(ret);

			// check various additional expectations
			long actualCompressionCount = (mode == ExecMode.HYBRID || mode == ExecMode.SINGLE_NODE) ? Statistics
				.getCPHeavyHitterCount("compress") : Statistics.getCPHeavyHitterCount("sp_compress");

			Assert.assertEquals(compressionCount, actualCompressionCount);
			if( compressionCount > 0 )
				Assert.assertTrue( mode == ExecMode.HYBRID ?
					heavyHittersContainsString("compress") : heavyHittersContainsString("sp_compress"));
			if( !testname.equals(TEST_NAME4) )
				Assert.assertFalse(heavyHittersContainsString("m_scale"));

		}
		catch(Exception e) {
			resetExecMode(oldPlatform);
			fail("Failed workload test");
		}
		finally {
			resetExecMode(oldPlatform);
			WorkloadAnalyzer.ALLOW_INTERMEDIATE_CANDIDATES = oldIntermediates;
		}
	}

	@Override
	protected File getConfigTemplateFile() {
		return new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-compress-workload.xml");
	}
}
