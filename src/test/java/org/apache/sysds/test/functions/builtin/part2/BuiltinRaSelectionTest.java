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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinRaSelectionTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "raSelection";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinRaSelectionTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"result"}));
	}

	@Test
	public void testRaSelectionTestGreaterThan() {
		//generate actual dataset and variables
		double[][] X = {
				{1.0, 2.0},
				{3.0, 4.0},
				{5.0, 6.0},
				{7.0, 8.0},
				{9.0, 10.0}};
		int select_col = 1;
		String op = Opcodes.GREATER.toString();
		double val = 4.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	@Test
	public void testRaSelectionGreaterThanOrEqul() {
		// Generate actual dataset and variables
		double[][] X = {
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0}
		};
		int select_col = 1;
		String op = Opcodes.GREATEREQUAL.toString();
		double val = 4.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	@Test
	public void testRaSelectionTestLessThan() {
		// Generate actual dataset and variables
		double[][] X = {
				{1.0, 2.0, 3.0, 4.0},
				{5.0, 6.0, 7.0, 8.0}
		};
		int select_col = 2;
		String op = Opcodes.LESS.toString();
		double val = 7.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	@Test
	public void testRaSelectionTestLessThanOrEqual() {
		// Generate actual dataset and variables
		double[][] X = {
				{5.0, 1.0, 3.0},
				{2.0, 4.0, 6.0},
				{7.0, 8.0, 9.0},
				{3.0, 5.0, 7.0},
				{1.0, 6.0, 8.0}
		};
		int select_col = 1;
		String op = Opcodes.LESSEQUAL.toString();
		double val = 4.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	@Test
	public void testRaSelectionTestEqual() {
		// Generate actual dataset and variables
		double[][] X = {
				{1.0, 2.0, 3.0, 4.0},
				{5.0, 6.0, 7.0, 8.0},
				{9.0, 10.0, 11.0, 12.0}
		};
		int select_col = 4;
		String op = Opcodes.EQUAL.toString();
		double val = 8.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	@Test
	public void testRaSelectionTestNotEqual() {
		// Generate actual dataset and variables
		double[][] X = {
				{1.0, 2.0, 3.0, 4.0},
				{5.0, 6.0, 7.0, 8.0},
				{9.0, 10.0, 11.0, 12.0},
				{13.0, 14.0, 15.0, 16.0}
		};
		int select_col = 2;
		String op = Opcodes.NOTEQUAL.toString();
		double val = 10.0;

		runRaSelectionTest(X, select_col, op, val);
	}

	private void runRaSelectionTest(double [][] X, int col, String op, double val)
	{
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args",
				input("X"), String.valueOf(col), op, String.valueOf(val), output("result") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " 
				+ inputDir() + " " + col + " " + op + " " + val + " " + expectedDir();

			writeInputMatrixWithMTD("X", X, true);
			//writeExpectedMatrix("result", Y);

			// run dmlScript and RScript
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("result");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("result");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Expected");
			
			//additional assertions
			if( !op.equals(Opcodes.EQUAL.toString()) )
				Assert.assertEquals(1, Statistics.getCPHeavyHitterCount(op));
			String otherOp = op.equals(Opcodes.NOTEQUAL.toString()) ? Opcodes.GREATER.toString() : Opcodes.NOTEQUAL.toString();
			Assert.assertFalse(heavyHittersContainsString(otherOp));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
