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

package org.apache.sysml.test.integration.functions.aggregate;

import java.io.IOException;
import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 */
public class FullGroupedAggregateMatrixTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "GroupedAggregateMatrix";
	private final static String TEST_NAME2 = "GroupedAggregateMatrixNoDims";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullGroupedAggregateMatrixTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1765;
	private final static int cols = 19;
	private final static int cols2 = 1007;
	
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private final static int numGroups = 17;
	
	private enum OpType{
		SUM,
		COUNT,
		MEAN,
		VARIANCE,
		MOMENT3,
		MOMENT4,
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"C"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"C"})); 
		TestUtils.clearAssertionInformation();

		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	//CP testcases

	@Test
	public void testGroupedAggSumDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3DenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3SparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4DenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4SparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, true, ExecType.CP);
	}

	//special CP testcases (negative)
	
	@Test
	public void testGroupedAggSumDenseCPNoDims() {
		runGroupedAggregateOperationTest(TEST_NAME2, OpType.SUM, false, ExecType.CP);
	}
	

	//SP testcases
	
	@Test
	public void testGroupedAggSumDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggSumSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggCountDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggCountSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, true, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, true, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggVarDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggVarSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, true, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment3DenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment3SparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, true, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment4DenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment4SparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, true, ExecType.SPARK);
	}

	@Test
	public void testGroupedAggSumDenseWideSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.SPARK, cols2);
	}
	
	@Test
	public void testGroupedAggSumSparseWideSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.SPARK, cols2);
	}

	
	//MR testcases
	
	@Test
	public void testGroupedAggSumDenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggSumSparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountDenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountSparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.COUNT, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanDenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanSparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MEAN, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggVarDenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggVarSparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.VARIANCE, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment3DenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment3SparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT3, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment4DenseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment4SparseMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MOMENT4, true, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggSumDenseWideMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.MR, cols2);
	}
	
	@Test
	public void testGroupedAggSumSparseWideMR() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.MR, cols2);
	}
	
	/**
	 * 
	 * @param testname
	 * @param type
	 * @param sparse
	 * @param instType
	 */
	private void runGroupedAggregateOperationTest( String testname, OpType type, boolean sparse, ExecType instType) {
		runGroupedAggregateOperationTest(testname, type, sparse, instType, cols);
	}
	
	/**
	 * 
	 * @param testname
	 * @param type
	 * @param sparse
	 * @param instType
	 */
	@SuppressWarnings("rawtypes")
	private void runGroupedAggregateOperationTest( String testname, OpType type, boolean sparse, ExecType instType, int numCols) 
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			//determine script and function name
			String TEST_NAME = testname;
			int fn = type.ordinal();
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			String TEST_CACHE_DIR = TEST_CACHE_ENABLED ? TEST_NAME + type.ordinal() + "_" + sparsity + "/" : "";
			boolean exceptionExpected = !TEST_NAME.equals(TEST_NAME1);
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), 
				String.valueOf(fn), String.valueOf(numGroups), output("C") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + fn + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, numCols, -0.05, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows, numCols,1000,1000);
			MapReduceTool.writeMetaDataFile(input("A.mtd"), ValueType.DOUBLE, mc1, OutputInfo.TextCellOutputInfo);
			double[][] B = TestUtils.round(getRandomMatrix(rows, 1, 1, numGroups, 1.0, 3)); 
			writeInputMatrix("B", B, true);
			MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,1,1000,1000);
			MapReduceTool.writeMetaDataFile(input("B.mtd"), ValueType.DOUBLE, mc2, OutputInfo.TextCellOutputInfo);
			
			//run tests
			Class cla = (exceptionExpected ? DMLException.class : null);
			runTest(true, exceptionExpected, cla, -1); 
			
			//compare matrices 
			if( !exceptionExpected ){
				runRScript(true); 
				
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
		}
		catch(IOException ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
		
}