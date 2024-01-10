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

package org.apache.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NaiveBayesTest  extends AutomatedTestBase{
	
	protected final static String TEST_DIR = "applications/naive-bayes/";
	protected final static String TEST_NAME = "naive-bayes";
	protected String TEST_CLASS_DIR = TEST_DIR + NaiveBayesTest.class.getSimpleName() + "/";

	protected int numRecords, numFeatures, numClasses;
	protected double sparsity;

	public NaiveBayesTest(int rows, int cols, int nc, double sp) {
		numRecords = rows;
		numFeatures = cols;
		numClasses = nc;
		sparsity = sp;
	}

	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //sparse tests (sparsity=0.01)
			   {100, 50, 10, 0.01}, // example running time: 3.5s (dml: .3s)
			   {1000, 500, 10, 0.01}, // example running time: 5s (dml: .8s)
			   {10000, 750, 10, 0.01}, // example running time: 32s (dml: .7s)
			   //{100000, 1000, 10, 0.01}, // example running time: 471s (dml: 3s)
			   //dense tests (sparsity=0.7)
			   {100, 50, 10, 0.7}, // example running time: 2s (dml: .2s)
			   {1000, 500, 10, 0.7}, // example running time: 6s (dml: .7s)
			   {10000, 750, 10, 0.7} // example running time: 61s (dml: 5.6s)
			   };
	   return Arrays.asList(data);
	 }
	 
	 @Override
	 public void setUp() {
		 addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	 }
	 
	 @Test
	 public void testNaiveBayes()
	 {
		 System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + numRecords + ", "
					+ numFeatures + ", " + numClasses + ", " + sparsity + "} ------------");
		 
		 int rows = numRecords;
		 int cols = numFeatures;
		 int classes = numClasses;
		 double sparsity = this.sparsity;
		 double laplace_correction = 1;
	        
		 getAndLoadTestConfiguration(TEST_NAME);
	     
		 List<String> proArgs = new ArrayList<>();
		 proArgs.add("-stats");
		 proArgs.add("-nvargs");
		 proArgs.add("X=" + input("X"));
		 proArgs.add("Y=" + input("Y"));
		 proArgs.add("classes=" + classes);
		 proArgs.add("laplace=" + laplace_correction);
		 proArgs.add("prior=" + output("prior"));
		 proArgs.add("conditionals=" + output("conditionals"));
		 proArgs.add("accuracy=" + output("accuracy"));
		 programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		 fullDMLScriptName = getScript();

		 rCmd = getRCmd(inputDir(), Integer.toString(classes), Double.toString(laplace_correction), expectedDir());
		 
		 double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		 double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		 for(int i=0; i<rows; i++){
			 Y[i][0] = (int)(Y[i][0]*classes) + 1;
			 Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
	     }	
	        
		 writeInputMatrixWithMTD("X", X, true);
		 writeInputMatrixWithMTD("Y", Y, true);
	        
		 runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
	        
		 runRScript(true);
	        
		 HashMap<CellIndex, Double> priorR = readRMatrixFromExpectedDir("prior");
		 HashMap<CellIndex, Double> priorSYSTEMDS= readDMLMatrixFromOutputDir("prior");
		 HashMap<CellIndex, Double> conditionalsR = readRMatrixFromExpectedDir("conditionals");
		 HashMap<CellIndex, Double> conditionalsSYSTEMDS = readDMLMatrixFromOutputDir("conditionals");
		 TestUtils.compareMatrices(priorR, priorSYSTEMDS, Math.pow(10, -12), "priorR", "priorSYSTEMDS");
		 TestUtils.compareMatrices(conditionalsR, conditionalsSYSTEMDS, Math.pow(10.0, -12.0), "conditionalsR", "conditionalsSYSTEMDS");
	 }
}
