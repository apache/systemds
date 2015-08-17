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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class MultiClassSVMTest  extends AutomatedTestBase{
	
	private final static String TEST_DIR = "applications/m-svm/";
	private final static String TEST_MULTICLASSSVM = "m-svm";

	private int numRecords, numFeatures, numClasses;
    private double sparsity;
    boolean intercept;
    
    public MultiClassSVMTest(int rows, int cols, int nc, boolean intercept, double sp) {
		numRecords = rows;
		numFeatures = cols;
		numClasses = nc;
		sparsity = sp;
		this.intercept = intercept;
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
		 setUpBase();
		 addTestConfiguration(TEST_MULTICLASSSVM, new TestConfiguration(TEST_DIR, "m-svm",
	                new String[] { "w" }));
	 }
	 
	 @Test
	 public void testMULTICLASSSVM()
	 {
		 int rows = numRecords;
		 int cols = numFeatures;
		 int classes = numClasses;
		 double sparsity = this.sparsity;
		 double tol = 0.001;
		 double reg = 1;
		 int maxiter = 100;
	        
		 TestConfiguration config = getTestConfiguration(TEST_MULTICLASSSVM);
	        
		 String MULTICLASSSVM_HOME = SCRIPT_DIR + TEST_DIR;
		 fullDMLScriptName = MULTICLASSSVM_HOME + TEST_MULTICLASSSVM + ".dml";
		 programArgs = new String[]{"-stats", "-nvargs", 
				 "X=" + MULTICLASSSVM_HOME + INPUT_DIR + "X", 
				 "Y=" + MULTICLASSSVM_HOME + INPUT_DIR + "Y",
				 "classes=" + Integer.toString(classes),
				 "tol=" + Double.toString(tol),
				 "reg=" + Double.toString(reg),
				 "maxiter=" + Integer.toString(maxiter),
				 "icpt=" + ((intercept) ? "1" : "0"),
				 "model=" + MULTICLASSSVM_HOME + OUTPUT_DIR + "w",
				 "Log=" + MULTICLASSSVM_HOME + OUTPUT_DIR + "Log"};
			
		 fullRScriptName = MULTICLASSSVM_HOME + TEST_MULTICLASSSVM + ".R";
		 rCmd = "Rscript" + " " + 
				 fullRScriptName + " " + 
				 MULTICLASSSVM_HOME + INPUT_DIR + " " + 
				 Integer.toString(classes) + " " + 
				 Double.toString(tol) + " " +
				 Double.toString(reg) + " " +
				 Integer.toString(maxiter) + " " +
				 ((intercept) ? "1" : "0") + " " +
				 MULTICLASSSVM_HOME + EXPECTED_DIR;
				
		 loadTestConfiguration(config);

		 double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		 double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		 for(int i=0; i<rows; i++){
			 Y[i][0] = (int)(Y[i][0]*classes) + 1;
			 Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
	     }	
	        
		 writeInputMatrixWithMTD("X", X, true);
		 writeInputMatrixWithMTD("Y", Y, true);
	        
		 runTest(true, false, null, -1);
	        
		 runRScript(true);
		 disableOutAndExpectedDeletion();
	        
		 HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
		 HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
		 boolean success = TestUtils.compareMatrices(wR, wDML, Math.pow(10, -10), "w", "w");
		 System.out.println(success+"");
	 }
}