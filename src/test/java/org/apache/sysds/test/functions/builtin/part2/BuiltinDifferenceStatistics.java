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

package org.apache.sysds.test.functions.builtin.part2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinDifferenceStatistics extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(BuiltinDifferenceStatistics.class.getName());

	private final static String TEST_NAME = "differenceStatistics";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDifferenceStatistics.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testDifferenceStatistics() {
		run(ExecType.CP, 0.1);
	}

	@Test
	public void testDifferenceStatisticsV2() {
		run(ExecType.CP, 0.2);
	}

	@Test
	public void testDifferenceStatisticsV3() {
		run(ExecType.CP, 0.01);
	}

	@Test
	public void testDifferenceStatisticsV4() {
		run(ExecType.CP, 0.001);
	}

	@Test
	public void testDifferenceStatisticsV5() {
		run(ExecType.CP, 0.4);
	}

	private void run(ExecType instType, double error) {

		ExecMode platformOld = setExecMode(instType);

		try {

			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), input("B")};

			MatrixBlock A = TestUtils.generateTestMatrixBlock(100, 5, -3, 3, 1.0, 1342);
			writeInputMatrixWithMTD("A", A, false);
			MatrixBlock C = TestUtils.generateTestMatrixBlock(1, 5, 1 - error, 1 + error, 1.0, 1342);
			MatrixBlock B = new MatrixBlock(100, 5, false);
			B = LibMatrixBincell.bincellOp(A, C, B, new BinaryOperator(Multiply.getMultiplyFnObject()), 1);
			writeInputMatrixWithMTD("B", B, true);
			String log = runTest(null).toString();
			// LOG.error(log);
			assertTrue(log.contains("Quantile Root Square Error"));
			double rmse = extractRootMeanSquareError(log);

			assertEquals(error, rmse, error * 0.01);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private double extractRootMeanSquareError(String log) {
		String[] lines = log.split("\n");
		for(String l : lines) {
			if(l.contains("Root Mean Square Error:")) {
				return Double.parseDouble(l.substring(35, l.length()));
			}
		}
		return 1;
	}
}
