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

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;


@RunWith(value = Parameterized.class)
public class ID3Test extends AutomatedTestBase
{
	
    private final static String TEST_DIR = "applications/id3/";
    private final static String TEST_NAME = "id3";

    private int numRecords, numFeatures;
    
	public ID3Test(int numRecords, int numFeatures) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   //TODO fix R script (values in 'nodes' for different settings incorrect, e.g., with minSplit=10 instead of 2)	 
	   Object[][] data = new Object[][] { {100, 50}, {1000, 50} };
	   return Arrays.asList(data); //10000
	 }

    @Override
    public void setUp()
    {
    	setUpBase();
    	addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "nodes", "edges"  }));
    }
    
    @Test
    public void testID3() throws IOException
    {
    	int rows = numRecords;			// # of rows in the training data 
        int cols = numFeatures;

        TestConfiguration config = getTestConfiguration(TEST_NAME);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-explain", "-args", HOME + INPUT_DIR + "X" , 
				                            HOME + INPUT_DIR + "y" ,
                                            Integer.toString(rows), 
				                            Integer.toString(cols),
				                            HOME + OUTPUT_DIR + "nodes",
				                            HOME + OUTPUT_DIR + "edges"};
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
      
        loadTestConfiguration(config);

        // prepare training data set
        double[][] X = round(getRandomMatrix(rows, cols, 1, 10, 1.0, 3));
        double[][] y = round(getRandomMatrix(rows, 1, 1, 10, 1.0, 7));
        writeInputMatrix("X", X, true);
        writeInputMatrix("y", y, true);
        
        //run tests
        //(changed expected MR from 62 to 66 because we now also count MR jobs in predicates)
        //(changed expected MR from 66 to 68 because we now rewrite sum(v1*v2) to t(v1)%*%v2 which rarely creates more jobs due to MMCJ incompatibility of other operations)
		runTest(true, false, null, 68); //max 68 compiled jobs		
		runRScript(true);
        
		//check also num actually executed jobs
		
		if(AutomatedTestBase.rtplatform != RUNTIME_PLATFORM.SPARK) {
			long actualMR = Statistics.getNoOfExecutedMRJobs();
			Assert.assertEquals("Wrong number of executed jobs: expected 0 but executed "+actualMR+".", 0, actualMR);
		}
		
		
		//compare results
        HashMap<CellIndex, Double> nR = readRMatrixFromFS("nodes");
        HashMap<CellIndex, Double> nDML= readDMLMatrixFromHDFS("nodes");
        HashMap<CellIndex, Double> eR = readRMatrixFromFS("edges");
        HashMap<CellIndex, Double> eDML= readDMLMatrixFromHDFS("edges");
        TestUtils.compareMatrices(nR, nDML, Math.pow(10, -14), "nR", "nDML");
        TestUtils.compareMatrices(eR, eDML, Math.pow(10, -14), "eR", "eDML");      
    }
    
    private double[][] round( double[][] data )
	{
		for( int i=0; i<data.length; i++ )
			for( int j=0; j<data[i].length; j++ )
				data[i][j] = Math.round(data[i][j]);
		return data;
	}
}
