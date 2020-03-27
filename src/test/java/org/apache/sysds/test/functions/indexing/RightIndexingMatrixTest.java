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

package org.apache.sysds.test.functions.indexing;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;


public class RightIndexingMatrixTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "RightIndexingMatrixTest";
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RightIndexingMatrixTest.class.getSimpleName() + "/";
	
	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	private final static double sparsity1 = 0.5;
	private final static double sparsity2 = 0.01;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
			new String[] {"B", "C", "D"}));
	}
	
	@Test
	public void testRightIndexingDenseCP() {
		runRightIndexingTest(ExecType.CP, false);
	}
	
	@Test
	public void testRightIndexingDenseSP() {
		runRightIndexingTest(ExecType.SPARK, false);
	}
	
	@Test
	public void testRightIndexingSparseCP() {
		runRightIndexingTest(ExecType.CP, true);
	}
	
	@Test
	public void testRightIndexingSparseSP() {
		runRightIndexingTest(ExecType.SPARK, true);
	}
	
	//various regression tests that led to test failures before
	
	@Test
	public void testRightIndexingDenseCPFixed1() {
		runRightIndexingTest(ExecType.CP, false, 2083, 2083, 437, 842);
	}
	
	@Test
	public void testRightIndexingDenseCPFixed2() {
		runRightIndexingTest(ExecType.CP, false, 1632, 1632, 282, 345);
	}
	
	public void runRightIndexingTest( ExecType et, boolean sparse ) {
		Random rand = new Random(System.currentTimeMillis());
		long rl = (long)(rand.nextDouble()*rows)+1;
		long ru = (long)(rand.nextDouble()*(rows-rl+1))+rl;
		long cl = (long)(rand.nextDouble()*cols)+1;
		long cu = (long)(rand.nextDouble()*(cols-cl+1))+cl;
		runRightIndexingTest(et, sparse, rl, ru, cl, cu);
	}
	
	public void runRightIndexingTest( ExecType et, boolean sparse, long rl, long ru, long cl, long cu )
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}	
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			double sparsity = sparse ? sparsity2 : sparsity1;
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			config.addVariable("rowstart", rl);
			config.addVariable("rowend", ru);
			config.addVariable("colstart", cl);
			config.addVariable("colend", cu);
			loadTestConfiguration(config);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args",  input("A"),
				Long.toString(rows), Long.toString(cols),
				Long.toString(rl), Long.toString(ru),
				Long.toString(cl), Long.toString(cu),
				output("B"), output("C"), output("D") };
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + rl + " " + ru + " " + cl + " " + cu + " " + expectedDir();
			
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
			writeInputMatrix("A", A, true);
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare results
			for(String file: config.getOutputFiles()) {
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
