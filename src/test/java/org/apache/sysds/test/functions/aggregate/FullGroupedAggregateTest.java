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

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 */
public class FullGroupedAggregateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "GroupedAggregate";
	private final static String TEST_NAME2 = "GroupedAggregateWeights";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullGroupedAggregateTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 17654;
	private final static int cols = 1;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private final static int numGroups = 7;
	private final static int maxWeight = 10;
	
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
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"D"})); 
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

	@Test
	public void testGroupedAggSumDenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggSumSparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, false, ExecType.SPARK);
	}
	
	// This is not applicable for spark instruction
//	@Test
//	public void testGroupedAggSumDenseTransposeSP() 
//	{
//		runGroupedAggregateOperationTest(OpType.SUM, false, false, true, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testGroupedAggSumSparseTransposeSP() 
//	{
//		runGroupedAggregateOperationTest(OpType.SUM, true, false, true, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testGroupedAggSumDenseWeightsTransposeSP() 
//	{
//		runGroupedAggregateOperationTest(OpType.SUM, false, true, true, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testGroupedAggSumSparseWeightsTransposeSP() 
//	{
//		runGroupedAggregateOperationTest(OpType.SUM, true, true, true, ExecType.SPARK);
//	}
	
	@Test
	public void testGroupedAggCountDenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggCountSparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggCountDenseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggCountSparseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanDenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanSparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanDenseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMeanSparseWeightsSP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggVarianceDenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggVarianceSparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, false, false, ExecType.SPARK);
	}
	
	
	@Test
	public void testGroupedAggMoment3DenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment3SparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, false, false, ExecType.SPARK);
	}
	
	
	@Test
	public void testGroupedAggMoment4DenseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testGroupedAggMoment4SparseSP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, false, false, ExecType.SPARK);
	}
	
	// -----------------------------------------------------------------------
	
	@Test
	public void testGroupedAggSumDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted variance in R
	@Test
	public void testGroupedAggVarianceDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, true, false, ExecType.CP);
	}
	*/
	
	@Test
	public void testGroupedAggMoment3DenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3SparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment3DenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3SparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, true, false, ExecType.CP);
	}
	*/
	
	@Test
	public void testGroupedAggMoment4DenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4SparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment4DenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4SparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, true, false, ExecType.CP);
	}
	*/
	
	private void runGroupedAggregateOperationTest( OpType type, boolean sparse, boolean weights, boolean transpose, ExecType instType) 
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			//determine script and function name
			String TEST_NAME = weights ? TEST_NAME2 : TEST_NAME1;
			int fn = type.ordinal();
			
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = TEST_NAME + type.ordinal() + "_" + sparsity + "_" + transpose + "/";
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			if( !weights ){
				programArgs = new String[]{"-explain","-args", input("A"), input("B"), 
					Integer.toString(fn), output("C") };
			}
			else{
				programArgs = new String[]{"-args", input("A"), input("B"), input("C"),
					Integer.toString(fn), output("D") };
			}
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + fn + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(transpose?cols:rows, transpose?rows:cols, -0.05, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
			MatrixCharacteristics mc1 = new MatrixCharacteristics(transpose?cols:rows, transpose?rows:cols,1000,1000);
			HDFSTool.writeMetaDataFile(input("A.mtd"), ValueType.FP64, mc1, FileFormat.TEXT);
			double[][] B = TestUtils.round(getRandomMatrix(rows, cols, 1, numGroups, 1.0, 3)); 
			writeInputMatrix("B", B, true);
			MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,cols,1000,1000);
			HDFSTool.writeMetaDataFile(input("B.mtd"), ValueType.FP64, mc2, FileFormat.TEXT);
			if( weights ){
				//currently we use integer weights due to our definition of weight as multiplicity
				double[][] C = TestUtils.round(getRandomMatrix(rows, cols, 1, maxWeight, 1.0, 3)); 
				writeInputMatrix("C", C, true);
				MatrixCharacteristics mc3 = new MatrixCharacteristics(rows,cols,1000,1000);
				HDFSTool.writeMetaDataFile(input("C.mtd"), ValueType.FP64, mc3, FileFormat.TEXT);	
			}
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(weights?"D":"C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(weights?"D":"C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
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