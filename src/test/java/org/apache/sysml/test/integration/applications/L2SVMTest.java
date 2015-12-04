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

public abstract class L2SVMTest extends AutomatedTestBase 
{
	
	protected final static String TEST_DIR = "applications/l2svm/";
	protected final static String TEST_NAME = "L2SVM";

	protected int numRecords, numFeatures;
	protected double sparsity;
	protected boolean intercept;
    
	public L2SVMTest(int rows, int cols, double sp, boolean intercept) {
		numRecords = rows;
		numFeatures = cols;
		sparsity = sp;
		intercept = this.intercept;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   //sparse tests (sparsity=0.01)
			   {100, 50, 0.01, false}, {1000, 500, 0.01, false}, {10000, 750, 0.01, false}, {10000, 750, 0.01, true}, {100000, 1000, 0.01, false},
			   //dense tests (sparsity=0.7)
			   {100, 50, 0.7, false}, {1000, 500, 0.7, false}, {1000, 500, 0.7, true}, {10000, 750, 0.7, false} };
	   return Arrays.asList(data);
	 }
	
	@Override
	public void setUp() {
    	addTestConfiguration(TEST_DIR, TEST_NAME);
	}
	
	protected void testL2SVM(ScriptType scriptType)
	{
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" + numRecords + ", " + numFeatures
				+ ", " + sparsity + ", " + intercept + "} ------------");
		this.scriptType = scriptType;
    	
		int rows = numRecords;
        int cols = numFeatures;
        double epsilon = 1.0e-8;
        double lambda = 1.0;
        int maxiterations = 3;
        int maxNumberOfMRJobs = 21;
        
        getAndLoadTestConfiguration(TEST_NAME);
        
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-stats");
		proArgs.add("-nvargs");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		proArgs.add("icpt=" + (intercept ? 1 : 0));
		proArgs.add("tol=" + epsilon);
		proArgs.add("reg=" + lambda);
		proArgs.add("maxiter=" + maxiterations);
		proArgs.add("model=" + output("w"));
		proArgs.add("Log=" + output("Log"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), (intercept ? Integer.toString(1) : Integer.toString(0)), Double.toString(epsilon), 
				Double.toString(lambda), Integer.toString(maxiterations), expectedDir());

        double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
        double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, -1);
        for(int i=0; i<rows; i++)
        	Y[i][0] = (Y[i][0] > 0) ? 1 : -1;
        
        writeInputMatrixWithMTD("X", X, true);
        writeInputMatrixWithMTD("Y", Y, true);
     
        runTest(true, EXCEPTION_NOT_EXPECTED, null, maxNumberOfMRJobs);
		
		runRScript(true);
        
		HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wSYSTEMML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wSYSTEMML, Math.pow(10, -12), "wR", "wSYSTEMML");
    }
}
