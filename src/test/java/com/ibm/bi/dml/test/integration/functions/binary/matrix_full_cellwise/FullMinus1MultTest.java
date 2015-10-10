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

package com.ibm.bi.dml.test.integration.functions.binary.matrix_full_cellwise;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class FullMinus1MultTest extends AutomatedTestBase 
{	
	private final static String TEST_NAME = "Minus1MultTest";
	private final static String TEST_DIR = "functions/binary/matrix_full_cellwise/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMinus1MultTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	
	private final static int rows = 3102;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.45;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testMinus1MultMatrixMatrixCP() {
		runMinus1MultTest(-1, ExecType.CP, true);
	}
	
	@Test
	public void testMinus1MultScalarMatrixCP() {
		runMinus1MultTest(1, ExecType.CP, true);
	}
	
	@Test
	public void testMinus1MultMatrixScalarCP() {
		runMinus1MultTest(2, ExecType.CP, true);
	}
	
	@Test
	public void testMinus1MultMatrixMatrixSP() {
		runMinus1MultTest(-1, ExecType.SPARK, true);
	}
	
	@Test
	public void testMinus1MultScalarMatrixSP() {
		runMinus1MultTest(1, ExecType.SPARK, true);
	}
	
	@Test
	public void testMinus1MultMatrixScalarSP() {
		runMinus1MultTest(2, ExecType.SPARK, true);
	}
	
	@Test
	public void testMinus1MultMatrixMatrixMR() {
		runMinus1MultTest(-1, ExecType.MR, true);
	}
	
	@Test
	public void testMinus1MultScalarMatrixMR() {
		runMinus1MultTest(1, ExecType.MR, true);
	}
	
	@Test
	public void testMinus1MultMatrixScalarMR() {
		runMinus1MultTest(2, ExecType.MR, true);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMinus1MultTest( int posScalar, ExecType instType, boolean rewrites)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		//rewrites
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			String TARGET_IN = TEST_DATA_DIR + TEST_CLASS_DIR + INPUT_DIR;
			String TARGET_OUT = TEST_DATA_DIR + TEST_CLASS_DIR + OUTPUT_DIR;
			String TARGET_EXPECTED = TEST_DATA_DIR + TEST_CLASS_DIR + EXPECTED_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args",  //stats required for opcode check
											TARGET_IN + "A",
											TARGET_IN + "B",
					                        TARGET_OUT + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       TARGET_IN + " " + TARGET_EXPECTED;
			
			loadTestConfiguration(config);
	
			//generate input datasets 
			double[][] A = getRandomMatrix((posScalar!=1)?rows:1, 1, -1, 1, (posScalar!=1)?sparsity1:1, 101); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix((posScalar!=2)?rows:1, 1, -1, 1, (posScalar!=2)?sparsity2:1, 102); 
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check generated opcode
			if( rewrites ){
				if( instType == ExecType.CP )
					Assert.assertTrue("Missing opcode: 1-*", Statistics.getCPHeavyHitterOpCodes().contains("1-*"));
				else if( instType == ExecType.SPARK )
					Assert.assertTrue("Missing opcode: sp_1-* | sp_map1-*", 
							Statistics.getCPHeavyHitterOpCodes().contains("sp_1-*") || 
							Statistics.getCPHeavyHitterOpCodes().contains("sp_map1-*"));
			}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}	
}