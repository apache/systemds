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

package org.apache.sysds.test.functions.append;

import java.util.HashMap;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class AppendVectorTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "AppendVectorTest";
	private final static String TEST_DIR = "functions/append/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AppendVectorTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int rows1 = 1279;
	private final static int cols1 = 1059;
	private final static int rows2 = 2021;
	private final static int cols2 = OptimizerUtils.DEFAULT_BLOCKSIZE;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"C"}));
	}
	
	@Test
	public void testAppendInBlockSP() {
		commonAppendTest(ExecMode.SPARK, rows1, cols1);
	}
	@Test
	public void testAppendOutBlockSP() {
		commonAppendTest(ExecMode.SPARK, rows2, cols2);
	}
	

	@Test
	public void testAppendInBlockCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows1, cols1);
	}
	
	@Test
	public void testAppendOutBlockCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows2, cols2);
	}	
	
	public void commonAppendTest(ExecMode platform, int rows, int cols)
	{
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
		ExecMode prevPlfm = setExecMode(platform);
		
		try {
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
	
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args",
					input("A"), Long.toString(rows), 
					Long.toString(cols), input("B"), output("C") };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
					inputDir() + " "+ expectedDir();
	
			Random rand=new Random(System.currentTimeMillis());
			double sparsity=rand.nextDouble();
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
			writeInputMatrix("A", A, true);
			sparsity=rand.nextDouble();
			double[][] B= getRandomMatrix(rows, 1, min, max, sparsity, System.currentTimeMillis());
			writeInputMatrix("B", B, true);
			
			boolean exceptionExpected = false;
			int numExpectedJobs = (platform == ExecMode.SINGLE_NODE) ? 0 : 4;
			
			runTest(true, exceptionExpected, null, numExpectedJobs);
			Assert.assertEquals("Wrong number of executed Spark jobs.",
				numExpectedJobs, Statistics.getNoOfExecutedSPInst());
		
			runRScript(true);
			
			for(String file: config.getOutputFiles()) {
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally {
			resetExecMode(prevPlfm);
		}
	}
}
