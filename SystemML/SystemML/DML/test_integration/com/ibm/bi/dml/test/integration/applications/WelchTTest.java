/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class WelchTTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/welchTTest/";
	private final static String TEST_WelchTTest = "welchTTest";
	
	private int numAttr, numPosSamples, numNegSamples;
	
	public WelchTTest(int numAttr, int numPosSamples, int numNegSamples){
		this.numAttr = numAttr;
		this.numPosSamples = numPosSamples;
		this.numNegSamples = numNegSamples;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { { 5, 100, 150}, { 50, 2000, 1500}, { 50, 7000, 1500}};
		return Arrays.asList(data);
	}
	 
	@Override
	public void setUp() {
		setUpBase();
    	addTestConfiguration(TEST_WelchTTest, 
    						 new TestConfiguration(TEST_DIR, 
    								 			   TEST_WelchTTest, 
    								 			   new String[] { "t_statistics", 
    								 							  "degrees_of_freedom" }));
	}
	
	@Test
	public void testWelchTTestWithRDMLAndJava() {
		TestConfiguration config = getTestConfiguration(TEST_WelchTTest);
		
		String WelchTTest_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = WelchTTest_HOME + TEST_WelchTTest + ".dml";
		
		programArgs = new String[]{"-args", WelchTTest_HOME + INPUT_DIR + "posSamples",
											WelchTTest_HOME + INPUT_DIR + "negSamples",
											WelchTTest_HOME + OUTPUT_DIR + "t_statistics",
											WelchTTest_HOME + OUTPUT_DIR + "degrees_of_freedom"};
		
		fullRScriptName = WelchTTest_HOME + TEST_WelchTTest + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " 
			   + WelchTTest_HOME + INPUT_DIR + " " 
			   + WelchTTest_HOME + EXPECTED_DIR;
	
		loadTestConfiguration(config);
		
		double[][] posSamples = getRandomMatrix(numPosSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());
		double[][] negSamples = getRandomMatrix(numNegSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());
		
		MatrixCharacteristics mc1 = new MatrixCharacteristics(numPosSamples,numAttr,-1,-1);
		writeInputMatrixWithMTD("posSamples", posSamples, true, mc1);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(numNegSamples,numAttr,-1,-1);
		writeInputMatrixWithMTD("negSamples", negSamples, true, mc2);
		
		boolean exceptionExpected = false;
		
		int expectedNumberOfJobs = 1;
		
		runTest(true, exceptionExpected, null, expectedNumberOfJobs); 
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		double tol = Math.pow(10, -13);
		HashMap<CellIndex, Double> t_statistics_R = readRMatrixFromFS("t_statistics");
        HashMap<CellIndex, Double> t_statistics_DML= readDMLMatrixFromHDFS("t_statistics");
        boolean success1 = TestUtils.compareMatrices(t_statistics_R, 
        											 t_statistics_DML, 
        											 tol, 
        											 "t_statistics_R", 
        											 "t_statistics_DML");
        
        HashMap<CellIndex, Double> degrees_of_freedom_R = readRMatrixFromFS("degrees_of_freedom");
        HashMap<CellIndex, Double> degrees_of_freedom_DML= readDMLMatrixFromHDFS("degrees_of_freedom");
        boolean success2 = TestUtils.compareMatrices(degrees_of_freedom_R, 
        											 degrees_of_freedom_DML, 
        											 tol, 
        											 "degrees_of_freedom_R", 
        											 "degrees_of_freedom_DML");		
	}
}
