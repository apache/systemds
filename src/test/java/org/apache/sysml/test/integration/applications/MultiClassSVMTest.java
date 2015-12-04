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

package org.apache.sysml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

public abstract class MultiClassSVMTest  extends AutomatedTestBase
{	
	protected final static String TEST_DIR = "applications/m-svm/";
	protected final static String TEST_NAME = "m-svm";

	protected int _numRecords;
	protected int _numFeatures;
	protected int _numClasses;
	protected double _sparsity;
	protected boolean _intercept;
    
    public MultiClassSVMTest(int rows, int cols, int nc, boolean intercept, double sp) {
		_numRecords = rows;
		_numFeatures = cols;
		_numClasses = nc;
		_intercept = intercept;
		_sparsity = sp;
	}
    
    @Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //sparse tests (sparsity=0.01)
			   {100, 50, 10, false, 0.01}, 
			   {1000, 500, 10, false, 0.01}, 
			   {1000, 500, 10, true, 0.01}, 
			   {10000, 750, 10, false, 0.01}, 
			   {10000, 750, 10, true, 0.01},
			   //dense tests (sparsity=0.7)
			   {100, 50, 10, false, 0.7}, 
			   {1000, 500, 10, false, 0.7}, 
			   {1000, 500, 10, true, 0.7}, 
			   {10000, 750, 10, false, 0.7} 
			   };
	   return Arrays.asList(data);
	 }
	 
	 @Override
	 public void setUp() {
		 addTestConfiguration(TEST_DIR, TEST_NAME);
	 }
	 
	 protected void testMultiClassSVM( ScriptType scriptType ) 
	 {
		 System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" +
				 _numRecords + ", " + 
				 _numFeatures + ", " + 
				 _numClasses + ", " + 
				 _intercept + ", " + 
				 _sparsity + "} ------------");
		 this.scriptType = scriptType;
		 
		 int rows = _numRecords;
		 int cols = _numFeatures;
		 int classes = _numClasses;
		 boolean intercept = _intercept;
		 double sparsity = _sparsity;
		 double tol = 0.001;
		 double reg = 1;
		 int maxiter = 100;
	     
		 getAndLoadTestConfiguration(TEST_NAME);
			
		 List<String> proArgs = new ArrayList<String>();
		 if (scriptType == ScriptType.PYDML) {
			 proArgs.add("-python");
		 }
		 proArgs.add("-stats");
		 proArgs.add("-nvargs");
		 proArgs.add("X=" + input("X"));
		 proArgs.add("Y=" + input("Y"));
		 proArgs.add("classes=" + classes);
		 proArgs.add("tol=" + tol);
		 proArgs.add("reg=" + reg);
		 proArgs.add("maxiter=" + maxiter);
		 proArgs.add("icpt=" + ((intercept) ? "1" : "0"));
		 proArgs.add("model=" + output("w"));
		 proArgs.add("Log=" + output("Log"));
		 programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		 //setup dml and R input arguments
		 fullDMLScriptName = getScript();
		 rCmd = getRCmd(inputDir(), Integer.toString(classes), Double.toString(tol), Double.toString(reg), 
				 Integer.toString(maxiter), ((intercept) ? "1" : "0"), expectedDir());

		 //generate input data
		 double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		 double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		 for(int i=0; i<rows; i++){
			 Y[i][0] = (int)(Y[i][0]*classes) + 1;
			 Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
	     }	
	        
		 //write input data and meta data files
		 writeInputMatrixWithMTD("X", X, true);
		 writeInputMatrixWithMTD("Y", Y, true);
	        
		 //run dml and R scripts
		 runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
	     runRScript(true);
	        
	     //compare outputs (assert on tear down)
		 HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
		 HashMap<CellIndex, Double> wSYSTEMML = readDMLMatrixFromHDFS("w");
		 TestUtils.compareMatrices(wR, wSYSTEMML, Math.pow(10, -10), "wR", "wSYSTEMML");
	 }
}