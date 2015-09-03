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

package com.ibm.bi.dml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.utils.TestUtils;

public abstract class MultiClassSVMTest  extends AutomatedTestBase{
	
	protected final static String TEST_DIR = "applications/m-svm/";
	protected final static String TEST_NAME = "m-svm";

	protected int numRecords, numFeatures, numClasses;
	protected double sparsity;
	protected boolean intercept;
    
    public MultiClassSVMTest(int rows, int cols, int nc, boolean intercept, double sp) {
		numRecords = rows;
		numFeatures = cols;
		numClasses = nc;
		this.intercept = intercept;
		sparsity = sp;
	}
    
    @Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //sparse tests (sparsity=0.01)
			   {100, 50, 10, false, 0.01}, 
			   {1000, 500, 10, false, 0.01}, 
			   {1000, 500, 10, true, 0.01}, 
			   {10000, 750, 10, false, 0.01}, 
			   {100000, 1000, 10, false, 0.01},
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
	 
	 protected void testMultiClassSVM(ScriptType scriptType) {
		 System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" +
				 numRecords + ", " + 
				 numFeatures + ", " + 
				 numClasses + ", " + 
				 intercept + ", " + 
				 sparsity + "} ------------");
		 this.scriptType = scriptType;
		 
		 int rows = numRecords;
		 int cols = numFeatures;
		 int classes = numClasses;
		 double sparsity = this.sparsity;
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
		 proArgs.add("classes=" + Integer.toString(classes));
		 proArgs.add("tol=" + Double.toString(tol));
		 proArgs.add("reg=" + Double.toString(reg));
		 proArgs.add("maxiter=" + Integer.toString(maxiter));
		 proArgs.add("icpt=" + ((intercept) ? "1" : "0"));
		 proArgs.add("model=" + output("w"));
		 proArgs.add("Log=" + output("Log"));
		 programArgs = proArgs.toArray(new String[proArgs.size()]);
		 System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
		 fullDMLScriptName = getScript();

		 rCmd = getRCmd(inputDir(), Integer.toString(classes), Double.toString(tol), Double.toString(reg), Integer.toString(maxiter), ((intercept) ? "1" : "0"), expectedDir());

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
	        
		 HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
		 HashMap<CellIndex, Double> wSYSTEMML= readDMLMatrixFromHDFS("w");
		 boolean success = TestUtils.compareMatrices(wR, wSYSTEMML, Math.pow(10, -10), "wR", "wSYSTEMML");
		 System.out.println(success);
	 }
}