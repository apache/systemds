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

public abstract class LinearLogRegTest extends AutomatedTestBase
{
	
    protected final static String TEST_DIR = "applications/linearLogReg/";
    protected final static String TEST_NAME = "LinearLogReg";

    protected int numRecords, numFeatures, numTestRecords;
    protected double sparsity;
    
	public LinearLogRegTest(int numRecords, int numFeatures, int numTestRecords, double sparsity) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
		this.numTestRecords = numTestRecords;
		this.sparsity = sparsity;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			 //sparse tests (sparsity=0.01)
			 {100, 50, 25, 0.01}, {1000, 500, 200, 0.01}, {10000, 750, 1500, 0.01}, {100000, 1000, 1500, 0.01},
			 //dense tests (sparsity=0.7)
			 {100, 50, 25, 0.7}, {1000, 500, 200, 0.7}, {10000, 750, 1500, 0.7}};
	   return Arrays.asList(data);
	 }
 
    @Override
    public void setUp()
    {
    	addTestConfiguration(TEST_DIR, TEST_NAME);
    }
    
    protected void testLinearLogReg(ScriptType scriptType) {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" + numRecords + ", " + numFeatures
				+ ", " + numTestRecords + ", " + sparsity + "} ------------");
		this.scriptType = scriptType;
    	
    	int rows = numRecords;			// # of rows in the training data 
        int cols = numFeatures;
        int rows_test = numTestRecords; // # of rows in the test data 
        int cols_test = cols;

        getAndLoadTestConfiguration(TEST_NAME);
           
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-stats");
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("Xt"));
		proArgs.add(input("y"));
		proArgs.add(input("yt"));
		proArgs.add(output("w"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
        
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), expectedDir());
		
        // prepare training data set
        double[][] X = getRandomMatrix(rows, cols, 1, 10, sparsity, 100);
        double[][] y = getRandomMatrix(rows, 1, 0.01, 1, 1, 100);
        writeInputMatrixWithMTD("X", X, true);
        writeInputMatrixWithMTD("y", y, true);

        // prepare test data set
        double[][] Xt = getRandomMatrix(rows_test, cols_test, 1, 10, sparsity, 100);
        double[][] yt = getRandomMatrix(rows_test, 1, 0.01, 1, 1, 100);
        writeInputMatrixWithMTD("Xt", Xt, true);
        writeInputMatrixWithMTD("yt", yt, true);
        
		int expectedNumberOfJobs = 31;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
        
		runRScript(true);

        HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wSYSTEMML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wSYSTEMML, Math.pow(10, -14), "wR", "wSYSTEMML");
    }
}
