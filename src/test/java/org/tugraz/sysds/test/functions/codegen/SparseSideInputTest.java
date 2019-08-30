/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.test.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class SparseSideInputTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "SparseSideInput";
	private static final String TEST_NAME1 = TEST_NAME+"1"; //row sum(X/rowSums(X)+Y)
	private static final String TEST_NAME2 = TEST_NAME+"2"; //cell sum(abs(X^2)+Y)
	private static final String TEST_NAME3 = TEST_NAME+"3"; //magg sum(X^2), sum(X+Y)
	private static final String TEST_NAME4 = TEST_NAME+"4"; //outer sum((X!=0) * (U %*% t(V) - Y))
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SparseSideInputTest.class.getSimpleName() + "/";
	private static String TEST_CONF1 = "SystemDS-config-codegen.xml";
	private static String TEST_CONF2 = "SystemDS-config-codegen-compress.xml";
	private static String TEST_CONF = TEST_CONF1;
	
	private static final int rows = 1798;
	private static final int cols = 784;
	private static final double sparsity = 0.1;
	private static final double eps = Math.pow(10, -7);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=4; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}
	
	@Test
	public void testCodegenRowULASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowCLASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowULASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowCLASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellULASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellCLASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellULASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellCLASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenMaggULASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenMaggCLASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenMaggULASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenMaggCLASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterULASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterCLASparseSideInputCP() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterULASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterCLASparseSideInputSP() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.SPARK );
	}
	
	private void testCodegenIntegration( String testname, boolean compress, ExecType instType )
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = rtplatform;
		switch( instType ) {
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			TEST_CONF = compress ? TEST_CONF2 : TEST_CONF1;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", 
				input("X"), input("Y"), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());
			
			//generate inputs
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			double[][] Y = getRandomMatrix(rows, cols, 0, 1, sparsity, 3);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			
			//run dml and r scripts
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoof") 
				|| heavyHittersContainsSubString("sp_spoof"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
			TEST_CONF = TEST_CONF2;
		}
	}
	
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		File f = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
		System.out.println("This test case overrides default configuration with " + f.getPath());
		return f;
	}
}
