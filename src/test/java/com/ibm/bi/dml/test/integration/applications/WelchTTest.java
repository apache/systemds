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
public class WelchTTest extends AutomatedTestBase {
	
	private final static String TEST_DIR = "applications/welchTTest/";
	private final static String TEST_WELCHTTEST = "welchTTest";
	private final static String WELCHTTEST_HOME = SCRIPT_DIR + TEST_DIR;
	
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
    	addTestConfiguration(TEST_WELCHTTEST, 
    						 new TestConfiguration(TEST_DIR, 
    								 TEST_WELCHTTEST, 
    								 			   new String[] { "t_statistics", 
    								 							  "degrees_of_freedom" }));
	}
	
	@Test
	public void testWelchTTestDml() {
		System.out.println("------------ BEGIN " + TEST_WELCHTTEST + " DML TEST {" + numAttr + ", " + numPosSamples + ", " + numNegSamples + "} ------------");
		testWelchTTest(ScriptType.DML);
	}

	@Test
	public void testWelchTTestPyDml() {
		System.out.println("------------ BEGIN " + TEST_WELCHTTEST + " PYDML TEST {" + numAttr + ", " + numPosSamples + ", " + numNegSamples + "} ------------");
		testWelchTTest(ScriptType.PYDML);
	}
	
	public void testWelchTTest(ScriptType scriptType) {
		this.scriptType = scriptType;
		
		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-args");
		proArgs.add(WELCHTTEST_HOME + INPUT_DIR + "posSamples");
		proArgs.add(WELCHTTEST_HOME + INPUT_DIR + "negSamples");
		proArgs.add(WELCHTTEST_HOME + OUTPUT_DIR + "t_statistics");
		proArgs.add(WELCHTTEST_HOME + OUTPUT_DIR + "degrees_of_freedom");
		
		switch (scriptType) {
		case DML:
			fullDMLScriptName = WELCHTTEST_HOME + TEST_WELCHTTEST + ".dml";
			break;
		case PYDML:
			fullPYDMLScriptName = WELCHTTEST_HOME + TEST_WELCHTTEST + ".pydml";
			proArgs.add(0, "-python");
			break;
		}
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
		fullRScriptName = WELCHTTEST_HOME + TEST_WELCHTTEST + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " 
			   + WELCHTTEST_HOME + INPUT_DIR + " " 
			   + WELCHTTEST_HOME + EXPECTED_DIR;
	
		TestConfiguration config = getTestConfiguration(TEST_WELCHTTEST);
		loadTestConfiguration(config);
		
		double[][] posSamples = getRandomMatrix(numPosSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());
		double[][] negSamples = getRandomMatrix(numNegSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());
		
		MatrixCharacteristics mc1 = new MatrixCharacteristics(numPosSamples,numAttr,-1,-1);
		writeInputMatrixWithMTD("posSamples", posSamples, true, mc1);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(numNegSamples,numAttr,-1,-1);
		writeInputMatrixWithMTD("negSamples", negSamples, true, mc2);
		
		int expectedNumberOfJobs = 1;
		
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs); 
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		double tol = Math.pow(10, -13);
		HashMap<CellIndex, Double> t_statistics_R = readRMatrixFromFS("t_statistics");
        HashMap<CellIndex, Double> t_statistics_DML= readDMLMatrixFromHDFS("t_statistics");
        TestUtils.compareMatrices(t_statistics_R, t_statistics_DML, tol, "t_statistics_R", "t_statistics_DML");
        
        HashMap<CellIndex, Double> degrees_of_freedom_R = readRMatrixFromFS("degrees_of_freedom");
        HashMap<CellIndex, Double> degrees_of_freedom_DML= readDMLMatrixFromHDFS("degrees_of_freedom");
        TestUtils.compareMatrices(degrees_of_freedom_R, degrees_of_freedom_DML, tol, "degrees_of_freedom_R", "degrees_of_freedom_DML");		
	}
}
