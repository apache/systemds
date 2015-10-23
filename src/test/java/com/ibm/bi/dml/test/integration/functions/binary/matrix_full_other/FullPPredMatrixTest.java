/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.BeforeClass;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * The main purpose of this test is to verify various input combinations for
 * matrix-matrix ppred operations that internally translate to binary operations.
 * 
 */
public class FullPPredMatrixTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "PPredMatrixTest";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullPPredMatrixTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1383;
	private final static int cols1 = 1432;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}
		
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) ); 
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
	
	@Test
	public void testPPredGreaterDenseDenseCP() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterDenseSparseCP() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterSparseDenseCP() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterSparseSparseCP() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsDenseDenseCP() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsDenseSparseCP() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsSparseDenseCP() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsSparseSparseCP() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsDenseDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsDenseSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsSparseDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsSparseSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessDenseDenseCP() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessDenseSparseCP() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessSparseDenseCP() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessSparseSparseCP() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsDenseDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsDenseSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsSparseDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsSparseSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}
	
	
	// ------------------------
	@Test
	public void testPPredGreaterDenseDenseSP() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterDenseSparseSP() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterSparseDenseSP() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterSparseSparseSP() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseDenseSP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseSparseSP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseDenseSP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseSparseSP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredEqualsDenseDenseSP() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredEqualsDenseSparseSP() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredEqualsSparseDenseSP() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredEqualsSparseSparseSP() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredNotEqualsDenseDenseSP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredNotEqualsDenseSparseSP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredNotEqualsSparseDenseSP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredNotEqualsSparseSparseSP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessDenseDenseSP() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessDenseSparseSP() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessSparseDenseSP() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessSparseSparseSP() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessEqualsDenseDenseSP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessEqualsDenseSparseSP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessEqualsSparseDenseSP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testPPredLessEqualsSparseSparseSP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.SPARK);
	}
	// ----------------------
	
	@Test
	public void testPPredGreaterDenseDenseMR() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterDenseSparseMR() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterSparseDenseMR() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterSparseSparseMR() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsDenseDenseMR() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsDenseSparseMR() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsSparseDenseMR() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsSparseSparseMR() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsDenseDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsDenseSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsSparseDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsSparseSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessDenseDenseMR() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessDenseSparseMR() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessSparseDenseMR() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessSparseSparseMR() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsDenseDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsDenseSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsSparseDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsSparseSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runPPredTest( Type type, boolean sp1, boolean sp2, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;
		    
	    RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
	    if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		double sparsityLeft = sp1 ? sparsity2 : sparsity1;
		double sparsityRight = sp2 ? sparsity2 : sparsity1;
		
		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = type.ordinal() + "_" + rows + "_" + cols + "_" + sparsityLeft + "_" + sparsityRight + "/";
		}
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			String TARGET_IN = TEST_DATA_DIR + TEST_CLASS_DIR + INPUT_DIR;
			String TARGET_OUT = TEST_DATA_DIR + TEST_CLASS_DIR + OUTPUT_DIR;
			String TARGET_EXPECTED = TEST_DATA_DIR + TEST_CLASS_DIR + EXPECTED_DIR + TEST_CACHE_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", TARGET_IN + "A",
                                                TARGET_IN + "B",
                                                Integer.toString(type.ordinal()),
                                                TARGET_OUT + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       TARGET_IN + " " + type.ordinal() + " " + TARGET_EXPECTED;
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsityLeft, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, cols, -15, 15, sparsityRight, 3); 
			writeInputMatrixWithMTD("B", B, true);
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}