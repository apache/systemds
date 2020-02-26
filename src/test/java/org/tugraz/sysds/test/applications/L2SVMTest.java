/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.applications;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.test.TestConstants.MatrixType;
import org.tugraz.sysds.test.TestConstants.SparsityType;

@RunWith(value = Parameterized.class)
public class L2SVMTest extends ApplicationTestBase {
	protected final static String TEST_DIR = "applications/l2svm/";
	private final static String TEST_CONF = "SystemDS-config-L2SVM.xml";
	private final static File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	protected final static String TEST_NAME = "L2SVM";
	protected String TEST_CLASS_DIR = TEST_DIR + L2SVMTest.class.getSimpleName() + "/" + Integer.toString(id) + "/";

	protected boolean intercept;

	protected double epsilon = 1e-10;
	protected double lambda = 1.0;
	protected int maxiterations = 3;
	protected int maxNumberOfMRJobs = 49;

	protected double[][] X;
	protected double[][] Y;

	List<String> proArgs;
	List<String> rArgs;

	public L2SVMTest(int id, SparsityType sparType, MatrixType matrixType, ExecMode newPlatform) {
		super( id, sparType, matrixType, newPlatform);
	}

	@Override
	public void setUp() {

		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		// addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);

		proArgs = Arrays.asList("-stats",
			"100",
			"-nvargs",
			"X=" + input("X"),
			"Y=" + input("Y"),
			"icpt=" + (intercept ? 1 : 0),
			"tol=" + epsilon,
			"reg=" + lambda,
			"maxiter=" + maxiterations,
			"model=" + output("w"),
			"Log=" + output("Log"));

		rArgs = Arrays.asList(inputDir(),
			(intercept ? Integer.toString(1) : Integer.toString(0)),
			Double.toString(epsilon),
			Double.toString(lambda),
			Integer.toString(maxiterations),
			expectedDir());

		X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		// 0.999 because sometimes the value is rounded up from 1.0000001 to 2.
		Y = TestUtils.round(getRandomMatrix(rows, 1, 0, 1, 1, 14));

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);

		platformOld = rtplatform;

		fullDMLScriptName = getScript();
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		rCmd = getRCmd(rArgs.toArray(new String[rArgs.size()]));

		runTest(true, EXCEPTION_NOT_EXPECTED, null, maxNumberOfMRJobs);

	}

	@After
	public void teardown() {
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	@Test
	public void testL2SVM() {
		runRScript(true);
		HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
		HashMap<CellIndex, Double> wSYSTEMDS = readDMLMatrixFromHDFS("w");
		TestUtils.compareMatrices(wR, wSYSTEMDS, epsilon, "wR", "wSYSTEMDS");
	}

	@Test
	public void testCompressedL2SVM() {
		boolean compressedHH = heavyHittersContainsSubString("compress");
		boolean sPCompressedHH =  heavyHittersContainsSubString("sp_compress");
		assertTrue("Compression was not a heavy hitter in execution " + this.toString() ,compressedHH || sPCompressedHH);
	}

	/**
	 * Override default configuration with custom test configuration to ensure scratch space and local temporary
	 * directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
