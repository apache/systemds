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

package org.apache.sysds.test.functions.aggregate;

import java.io.IOException;
import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

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
		MIN,
		MAX
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

	@Test
	public void testGroupedAggMinDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MIN, false, ExecType.CP);
	}

	@Test
	public void testGroupedAggMinSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MIN, true, ExecType.CP);
	}

	@Test
	public void testGroupedAggMaxDenseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MAX, false, ExecType.CP);
	}

	@Test
	public void testGroupedAggMaxSparseCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MAX, true, ExecType.CP);
	}

	@Test
	public void testGroupedAggSumDenseWideCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.CP, cols2);
	}
	
	@Test
	public void testGroupedAggSumSparseWideCP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.CP, cols2);
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
	public void testGroupedAggMinDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MIN, false, ExecType.SPARK);
	}

	@Test
	public void testGroupedAggMinSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MIN, true, ExecType.SPARK);
	}

	@Test
	public void testGroupedAggMaxDenseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MAX, false, ExecType.SPARK);
	}

	@Test
	public void testGroupedAggMaxSparseSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.MAX, true, ExecType.SPARK);
	}

	@Test
	public void testGroupedAggSumDenseWideSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, false, ExecType.SPARK, cols2);
	}
	
	@Test
	public void testGroupedAggSumSparseWideSP() {
		runGroupedAggregateOperationTest(TEST_NAME1, OpType.SUM, true, ExecType.SPARK, cols2);
	}

	private void runGroupedAggregateOperationTest( String testname, OpType type, boolean sparse, ExecType instType) {
		runGroupedAggregateOperationTest(testname, type, sparse, instType, cols);
	}
	
	@SuppressWarnings("rawtypes")
	private void runGroupedAggregateOperationTest( String testname, OpType type, boolean sparse, ExecType instType, int numCols) 
	{
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			//determine script and function name
			String TEST_NAME = testname;
			int fn = type.ordinal();
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			String TEST_CACHE_DIR = TEST_CACHE_ENABLED ? TEST_NAME + type.ordinal() + "_" + sparsity + "_" + numCols + "/" : "";
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
			HDFSTool.writeMetaDataFile(input("A.mtd"), ValueType.FP64, mc1, FileFormat.TEXT);
			double[][] B = TestUtils.round(getRandomMatrix(rows, 1, 1, numGroups, 1.0, 3)); 
			writeInputMatrix("B", B, true);
			MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,1,1000,1000);
			HDFSTool.writeMetaDataFile(input("B.mtd"), ValueType.FP64, mc2, FileFormat.TEXT);
			
			//run tests
			Class cla = (exceptionExpected ? DMLRuntimeException.class : null);
			runTest(true, exceptionExpected, cla, -1); 
			
			//compare matrices 
			if( !exceptionExpected ){
				//run R script for comparison
				runRScript(true); 
				
				//compare output matrices
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
				
				//check dml output meta data
				checkDMLMetaDataFile("C", new MatrixCharacteristics(numGroups,numCols,1,1));
			}
		}
		catch(IOException ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
