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
import org.junit.Ignore;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.util.concurrent.ThreadLocalRandom;

public class BuiltinOutlierByIQRTest extends AutomatedTestBase {
	private final static String TEST_NAME = "outlier_by_IQR";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierByIQRTest.class.getSimpleName() + "/";

	private final static int rows = 100;
	private final static int cols = 15;
	private final static double spDense = 0.7;
	private final static double spSparse = 0.8;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Ignore
	public void testOutlierRepair0CP() {
		runOutlierTest(false, 1.5, 0, 10,ExecType.CP);
	}

	@Test
	public void testOutlierRepair1CP() {
		runOutlierTest(false, 2, 1, 10,ExecType.CP);
	}


	@Test
	public void testOutlierRepair0SP() {
		runOutlierTest(false, 2, 0, 10,ExecType.SPARK);
	}

	@Test
	public void testOutlierRepair1SP() {
		runOutlierTest(false, 1.5, 1, 10,ExecType.SPARK);
	}
	@Test
	public void testOutlierRepair0IterativeCP() {
		runOutlierTest(false, 1.5, 0, 0,ExecType.CP);
	}

	@Test
	public void testOutlierRepair1IterativeCP() {
		runOutlierTest(false, 1.5, 1, 0,ExecType.CP);
	}


	@Test
	public void testOutlierRepair0IterativeSP() {
		runOutlierTest(false, 1.5, 0, 0,ExecType.SPARK);
	}

	@Test
	public void testOutlierRepair1IterativeSP() {
		runOutlierTest(false, 1.5, 1, 0,ExecType.SPARK);
	}

	@Test
	public void testOutlierRepair2IterativeCP() {
		runOutlierTest(false, 1.5, 2, 0,ExecType.CP);
	}

	@Test
	public void testOutlierRepair2IterativeSP() {
		runOutlierTest(false, 1.5, 2, 0,ExecType.SPARK);
	}

	private void runOutlierTest(boolean sparse, double  k,  int repair, int max_iterations, ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			setOutputBuffering(true);
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("A"), String.valueOf(k),
					 String.valueOf(repair), String.valueOf(max_iterations),output("B")};

			//generate actual dataset
			double[][] A =  getRandomMatrix(rows, cols, 1, 100, sparse?spSparse:spDense, 10);
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
