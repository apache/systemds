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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.util.concurrent.ThreadLocalRandom;

public class BuiltinOutlierBySDTest extends AutomatedTestBase {
	private final static String TEST_NAME = "outlier_by_sd";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierBySDTest.class.getSimpleName() + "/";

	private final static int rows = 50;
	private final static int cols = 10;
	private final static double spDense = 0.6;
	private final static double spSparse = 0.4;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testOutlierRepair0CP() {
		runOutlierTest(false, 2, 0, 0, ExecType.CP);
	}

	@Test
	public void testOutlierRepair1CP() {
		runOutlierTest(false, 2, 1, 0, ExecType.CP);
	}

	@Test
	public void testOutlierRepair2CP() {
		runOutlierTest(false, 2, 2, 10, ExecType.CP);
	}

	@Test
	public void testOutlierRepair2SP() {
		runOutlierTest(false, 2, 2, 0, ExecType.CP);
	}

	@Test
	public void testOutlierRepair0SP() {
		runOutlierTest(false, 2, 0, 10, ExecType.SPARK);
	}

	@Test
	public void testOutlierRepair1SP() {
		runOutlierTest(false, 2, 1, 10, ExecType.SPARK);
	}

	@Test
	public void testOutlierK3CP() {
		runOutlierTest(true, 3, 1, 10,ExecType.CP);
	}

	@Test
	public void testOutlierIterativeCP() {
		runOutlierTest(false, 2, 1, 0, ExecType.CP);
	}

	@Test
	public void testOutlierIterativeSP() {
		runOutlierTest(false, 2, 1, 10, ExecType.SPARK);
	}

	private void runOutlierTest(boolean sparse, double  k,  int repair, int max_iterations, ExecType instType)
	{
		setOutputBuffering(true);
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("A"), String.valueOf(k),
					String.valueOf(repair), String.valueOf(max_iterations),output("B")};

			//generate actual dataset
			double[][] A =  getRandomMatrix(rows, cols, 1, 10000, sparse?spSparse:spDense, 7);
			for(int i=0; i<A.length/4; i++) {
				int r = ThreadLocalRandom.current().nextInt(0, A.length);
				int c = ThreadLocalRandom.current().nextInt(0, A[0].length);
				double badValue = ThreadLocalRandom.current().nextDouble(0, A.length*100);
				A[r][c] = badValue;
			}

			writeInputMatrixWithMTD("A", A, true);

//			runTest(true, false, null, -1);
			String out = runTest(null).toString();
			Assert.assertTrue(out.contains("TRUE"));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
