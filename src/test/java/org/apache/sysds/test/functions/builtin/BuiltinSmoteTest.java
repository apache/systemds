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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinSmoteTest extends AutomatedTestBase {

	private final static String TEST_NAME = "smote";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSmoteTest.class.getSimpleName() + "/";

	private final static int rows = 20;
	private final static int colsX = 20;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

	@Test
	public void testSmote0CP() {
		runSmoteTest(100, 1, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote1CP() {
		runSmoteTest(300, 10, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote2CP() {
		runSmoteTest(400, 5, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote1Spark() {
		runSmoteTest(300, 3, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testSmote2Spark() { runSmoteTest(400, 5, LopProperties.ExecType.SPARK);	}


	private void runSmoteTest(int sample, int nn, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "X=" + input("X"), "S=" + sample, "K=" + nn , "Z="+output("Sum"), "T="+input("T")};

			double[][] X = getRandomMatrix(rows, colsX, 0, 1, 0.3, 1);

			writeInputMatrixWithMTD("X", X, true);

			double[][] T = getRandomMatrix(rows, colsX, 2, 3.0, 0.3, 3);

			writeInputMatrixWithMTD("T", T, true);

			runTest(true, false, null, -1);
			HashMap<MatrixValue.CellIndex, Double> value = readDMLMatrixFromHDFS("Sum");
			Assert.assertEquals("synthetic samples does not fall into minority class cluster",1,
				value.get(new MatrixValue.CellIndex(1,1)), 0.000001);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}
}

