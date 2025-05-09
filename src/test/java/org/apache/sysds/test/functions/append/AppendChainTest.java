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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class AppendChainTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "AppendChainTest";
	private final static String TEST_DIR = "functions/append/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AppendChainTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int min=1;
	private final static int max=100;
	
	private final static int rows = 492;
	private final static int cols1 = 1059;
	private final static int cols2a = 1;
	private final static int cols3a = 1;
	private final static int cols2b = 1030;
	private final static int cols3b = 1770;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"C"}));
	}

	@Test
	public void testAppendChainVectorDenseCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows, cols1, cols2b, cols3b, false);
	}
	
	// ------------------------------------------------------
	@Test
	public void testAppendChainVectorDenseSP() {
		commonAppendTest(ExecMode.SPARK, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseSP() {
		commonAppendTest(ExecMode.SPARK, rows, cols1, cols2b, cols3b, false);
	}
	
	@Test
	public void testAppendChainVectorSparseSP() {
		commonAppendTest(ExecMode.SPARK, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseSP() {
		commonAppendTest(ExecMode.SPARK, rows, cols1, cols2b, cols3b, true);
	}
	
	// ------------------------------------------------------
	
	@Test
	public void testAppendChainVectorSparseCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseCP() {
		commonAppendTest(ExecMode.SINGLE_NODE, rows, cols1, cols2b, cols3b, true);
	}
	
	public void commonAppendTest(ExecMode platform, int rows, int cols1, int cols2, int cols3, boolean sparse)
	{
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
		ExecMode prevPlfm=setExecMode(platform);
		
		try
		{
			config.addVariable("rows", rows);
			config.addVariable("cols", cols1);
	
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  input("A"), 
				Long.toString(rows), Long.toString(cols1),
				input("B1"), Long.toString(cols2), input("B2"),
				Long.toString(cols3), output("C") };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " "+ expectedDir();
	
			double sparsity = sparse ? sparsity2 : sparsity1; 
			double sparsity2 = 1-sparsity;
			double[][] A = getRandomMatrix(rows, cols1, min, max, sparsity, 11);
			writeInputMatrix("A", A, true);
			double[][] B1= getRandomMatrix(rows, cols2, min, max, sparsity2, 21);
			writeInputMatrix("B1", B1, true);
			double[][] B2= getRandomMatrix(rows, cols2, min, max, sparsity, 31);
			writeInputMatrix("B2", B2, true);
	
			int expectedCompiled = platform==ExecMode.SINGLE_NODE ?
				0 : 5; //3x(rblk), append, write
			
			runTest(true, false, null, expectedCompiled);
			runRScript(true);
			
			Assert.assertEquals("Wrong number of executed Spark jobs.",
				expectedCompiled, Statistics.getNoOfExecutedSPInst());
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
