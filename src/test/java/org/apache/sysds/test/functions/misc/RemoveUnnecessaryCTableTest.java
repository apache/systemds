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

package org.apache.sysds.test.functions.misc;

import org.junit.Test;
import org.junit.Assert;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;

import java.util.ArrayList;
import java.util.Arrays;

public class RemoveUnnecessaryCTableTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteRemoveUnnecessaryCTable1L";
	private static final String TEST_NAME2 = "RewriteRemoveUnnecessaryCTable1R";
	private static final String TEST_NAME3 = "RewriteRemoveUnnecessaryCTableTest";
	private static final String TEST_NAME4 = "RewriteRemoveUnnecessaryCTableTestTernary";
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RemoveUnnecessaryCTableTest.class.getSimpleName() + "/";
	private static final double[][] A = {{1},{2},{3},{4},{5},{6},{1},{2},{3},{4},{5},{6}};
	private static final double[][] ATransposed = {{1,2,3,4,5,6,7,8,9,10,11,12}};
	private static final double[][] AMultiColMultiRow = {{1,2,3},{4,5,6},{7,8,9}};

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "s" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "s" }) );
	}

	@Test
	public void testRemoveCTable1L() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME1, A, sum, true);
	}

	@Test
	public void testRemoveCTable1LTransposed() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME1, ATransposed, sum, true);
	}

	@Test
	public void testRemoveCTable1LMultiColMultiRow() {
		double[][] sum = {{9}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME1, AMultiColMultiRow, sum, true);
	}

	@Test
	public void testRemoveCTable1R() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME2, A, sum, false);
	}

	@Test
	public void testRemoveCTable1RTransposed() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME2, ATransposed, sum, false);
	}

	@Test
	public void testRemoveCTable1RMultiColMultiRow() {
		double[][] sum = {{9}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME2, AMultiColMultiRow, sum, false);
	}

	@Test
	public void testRemoveCTableSameA() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME3, A, A, sum, true);
	}

	@Test
	public void testRemoveCTableSameATransposed() {
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME3, ATransposed, ATransposed, sum, true);
	}

	@Test
	public void testRemoveCTableSameAMultiColMultiRow() {
		double[][] sum = {{9}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME3, AMultiColMultiRow, AMultiColMultiRow, sum, true);
	}

	@Test
	public void testRemoveCTableB() {
		double[][] B = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}};
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME3, A, B, sum, true);
	}

	@Test
	public void testRemoveCTableBTransposed() {
		double[][] B = {{1,2,3,4,5,6,7,8,9,10,11,12}};
		double[][] sum = {{12}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME3, ATransposed, B, sum, true);
	}

	@Test
	public void testNotRemoveCTableTernary() {
		double[][] B = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}};
		double[][] C = {{10},{10},{10},{10},{10},{10},{10},{10},{10},{10},{10},{10}};
		double[][] sum = {{120}};
		testRewriteRemoveUnnecessaryCTable(TEST_NAME4, A, B, C, sum, false);
	}

	private void testRewriteRemoveUnnecessaryCTable(String test, double[][] A, double[][] B, double[][] C, double[][] sum, boolean checkHeavyHitters){
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try
		{
			TestConfiguration config = getTestConfiguration(test);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + test + ".dml";

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = true;

			ArrayList<String> programArgsBuilder = new ArrayList<>(
				Arrays.asList("-stats", "-args" ));
			// Get Matrix Input
			if (A != null){
				programArgsBuilder.add(input("A"));
				writeInputMatrixWithMTD("A", A, true);
			}
			if (B != null) {
				programArgsBuilder.add(input("B"));
				writeInputMatrixWithMTD("B", B, true);
			}
			if (C != null){
				programArgsBuilder.add(input("C"));
				writeInputMatrixWithMTD("C", C, true);
			}

			programArgsBuilder.add(output(config.getOutputFiles()[0]));

			String[] argsArray = new String[programArgsBuilder.size()];
			argsArray = programArgsBuilder.toArray(argsArray);
			programArgs =  argsArray;
			
			//run test
			runTest(true, false, null, -1);
			
			//compare scalar
			double s = readDMLScalarFromHDFS("s").get(new CellIndex(1, 1));
			TestUtils.compareScalars(s,sum[0][0],1e-10);

			if( checkHeavyHitters ) {
				boolean table = heavyHittersContainsSubString("table");
				Assert.assertFalse("Heavy hitters should not contain table", table);
			}
		} 
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
	
	private void testRewriteRemoveUnnecessaryCTable(String test, double[][] A, double[][] sum, boolean checkHeavyHitters) {
		testRewriteRemoveUnnecessaryCTable(test, A, null, null, sum, checkHeavyHitters);
	}

	private void testRewriteRemoveUnnecessaryCTable(String test, double[][] A, double[][] B, double[][] sum, boolean checkHeavyHitters){
		testRewriteRemoveUnnecessaryCTable(test, A, B, null, sum, checkHeavyHitters);
	}
}
