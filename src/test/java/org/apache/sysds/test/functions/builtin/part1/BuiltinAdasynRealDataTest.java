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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinAdasynRealDataTest extends AutomatedTestBase {
	private final static String TEST_NAME = "adasynRealData";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinAdasynRealDataTest.class.getSimpleName() + "/";

	private final static String DIABETES_DATA = DATASET_DIR + "diabetes/diabetes.csv";
	private final static String DIABETES_TFSPEC = DATASET_DIR + "diabetes/tfspec.json";
	private final static String TITANIC_DATA = DATASET_DIR + "titanic/titanic.csv";
	private final static String TITANIC_TFSPEC = DATASET_DIR + "titanic/tfspec.json";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testDiabetesNoAdasyn() {
		runAdasynTest(DIABETES_DATA, DIABETES_TFSPEC, false, 0.783, -1, ExecType.CP);
	}
	
	@Test
	public void testDiabetesAdasynK4() {
		runAdasynTest(DIABETES_DATA, DIABETES_TFSPEC, true, 0.787, 4, ExecType.CP);
	}
	
	@Test
	public void testDiabetesAdasynK6() {
		runAdasynTest(DIABETES_DATA, DIABETES_TFSPEC, true, 0.787, 6, ExecType.CP);
	}
	
	@Test
	public void testTitanicNoAdasyn() {
		runAdasynTest(TITANIC_DATA, TITANIC_TFSPEC, false, 0.781, -1, ExecType.CP);
	}
	
	@Test
	public void testTitanicAdasynK4() {
		runAdasynTest(TITANIC_DATA, TITANIC_TFSPEC, true, 0.797, 4, ExecType.CP);
	}
	
	@Test
	public void testTitanicAdasynK5() {
		runAdasynTest(TITANIC_DATA, TITANIC_TFSPEC, true, 0.786, 5, ExecType.CP);
	}
	
	private void runAdasynTest(String data, String tfspec, boolean adasyn, double minAcc, int k, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args",
				data, tfspec, String.valueOf(adasyn), String.valueOf(k), output("R")};

			runTest(true, false, null, -1);

			double acc = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));

			Assert.assertTrue("Accuracy should be greater than min: " + acc + "  min: " + minAcc, acc >= minAcc);
			Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
