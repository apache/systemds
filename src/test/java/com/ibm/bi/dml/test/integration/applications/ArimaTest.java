/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

@RunWith(value = Parameterized.class)
public class ArimaTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/arima_box-jenkins/";
	private final static String TEST_ArimaTest = "arima";
	
	private int max_func_invoc, p, d, q, P, D, Q, s, include_mean, useJacobi;
	
	public ArimaTest(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi){
		this.max_func_invoc = m;
		this.p = p;
		this.d = d;
		this.q = q;
		this.P = P;
		this.D = D;
		this.Q = Q;
		this.s = s;
		this.include_mean = include_mean;
		this.useJacobi = useJacobi;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { //{ 10, 1, 0, 1, 0, 0, 0, 24, 0, 0}, 
										   { 10, 1, 1, 1, 1, 1, 1, 24, 1, 1}};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() {
		setUpBase();
    	addTestConfiguration(TEST_ArimaTest, 
    						 new TestConfiguration(TEST_DIR, 
    								 			   TEST_ArimaTest, 
    								 			   new String[] { "learnt.model"}));
	}
	
	@Test
	public void testArimaWithRDMLAndJava() {
		TestConfiguration config = getTestConfiguration(TEST_ArimaTest);
		
		String ArimaTest_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = ArimaTest_HOME + TEST_ArimaTest + ".dml";
		
		programArgs = new String[]{"-args", ArimaTest_HOME + INPUT_DIR + "col.mtx",
											""+max_func_invoc, ""+p, ""+d, ""+q, 
											""+P, ""+D, ""+Q, ""+s, 
											""+include_mean, ""+useJacobi,
											ArimaTest_HOME + OUTPUT_DIR + "learnt.model"};
		
		fullRScriptName = ArimaTest_HOME + TEST_ArimaTest + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " 
			   + ArimaTest_HOME + INPUT_DIR 
			   + " " + max_func_invoc + " " + p + " " + d + " " + q
			   + " " + P + " " + D + " " + Q + " " + s 
			   + " " + include_mean + " " + useJacobi
			   + " " + ArimaTest_HOME + EXPECTED_DIR;
	
		loadTestConfiguration(config);
		
		int timeSeriesLength = 5000;
		double[][] timeSeries = getRandomMatrix(timeSeriesLength, 1, 1, 5, 0.9, System.currentTimeMillis());
		
		MatrixCharacteristics mc = new MatrixCharacteristics(timeSeriesLength,1,-1,-1);
		writeInputMatrixWithMTD("col", timeSeries, true, mc);
		
		boolean exceptionExpected = false;
		
		runTest(true, exceptionExpected, null, -1);
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		double tol = Math.pow(10, -14);
		HashMap<CellIndex, Double> arima_model_R = readRMatrixFromFS("learnt.model");
        HashMap<CellIndex, Double> arima_model_DML= readDMLMatrixFromHDFS("learnt.model");
        TestUtils.compareMatrices(arima_model_R, arima_model_DML, tol, "arima_model_R", "arima_model_DML");
	}   
}
