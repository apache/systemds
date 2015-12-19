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

import org.junit.Assert;
import org.junit.runners.Parameterized.Parameters;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public abstract class ID3Test extends AutomatedTestBase
{
	
    protected final static String TEST_DIR = "applications/id3/";
    protected final static String TEST_NAME = "id3";
    protected String TEST_CLASS_DIR = TEST_DIR + ID3Test.class.getSimpleName() + "/";

    protected int numRecords, numFeatures;
    
	public ID3Test(int numRecords, int numFeatures) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   //TODO fix R script (values in 'nodes' for different settings incorrect, e.g., with minSplit=10 instead of 2)	 
	   Object[][] data = new Object[][] { {100, 50}, {1000, 50} };
	   return Arrays.asList(data);
	 }

    @Override
    public void setUp()
    {
    	addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
    }
    
    protected void testID3(ScriptType scriptType) 
    {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST {" + numRecords + ", "
				+ numFeatures + "} ------------");
		this.scriptType = scriptType;    	
    	
    	int rows = numRecords;			// # of rows in the training data 
        int cols = numFeatures;
        
        getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-explain");
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("y"));
		proArgs.add(output("nodes"));
		proArgs.add(output("edges"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), expectedDir());

        // prepare training data set
        double[][] X = round(getRandomMatrix(rows, cols, 1, 10, 1.0, 3));
        double[][] y = round(getRandomMatrix(rows, 1, 1, 10, 1.0, 7));
        writeInputMatrixWithMTD("X", X, true);
        writeInputMatrixWithMTD("y", y, true);
        
        //run tests
        //(changed expected MR from 62 to 66 because we now also count MR jobs in predicates)
        //(changed expected MR from 66 to 68 because we now rewrite sum(v1*v2) to t(v1)%*%v2 which rarely creates more jobs due to MMCJ incompatibility of other operations)
		runTest(true, EXCEPTION_NOT_EXPECTED, null, 70); //max 68 compiled jobs		
		runRScript(true);
        
		//check also num actually executed jobs
		if(AutomatedTestBase.rtplatform != RUNTIME_PLATFORM.SPARK) {
			long actualMR = Statistics.getNoOfExecutedMRJobs();
			Assert.assertEquals("Wrong number of executed jobs: expected 0 but executed "+actualMR+".", 0, actualMR);
		}
				
		//compare results
        HashMap<CellIndex, Double> nR = readRMatrixFromFS("nodes");
        HashMap<CellIndex, Double> nSYSTEMML= readDMLMatrixFromHDFS("nodes");
        HashMap<CellIndex, Double> eR = readRMatrixFromFS("edges");
        HashMap<CellIndex, Double> eSYSTEMML= readDMLMatrixFromHDFS("edges");
        TestUtils.compareMatrices(nR, nSYSTEMML, Math.pow(10, -14), "nR", "nSYSTEMML");
        TestUtils.compareMatrices(eR, eSYSTEMML, Math.pow(10, -14), "eR", "eSYSTEMML");      
    }
    
    private double[][] round( double[][] data )
	{
		for( int i=0; i<data.length; i++ )
			for( int j=0; j<data[i].length; j++ )
				data[i][j] = Math.round(data[i][j]);
		return data;
	}
}
