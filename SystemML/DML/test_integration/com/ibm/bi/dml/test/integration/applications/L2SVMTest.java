/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

import java.util.HashMap;


public class L2SVMTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/l2svm/";
	private final static String TEST_L2SVM = "L2SVM";

	@Override
	public void setUp() {
		setUpBase();
    	addTestConfiguration(TEST_L2SVM, new TestConfiguration(TEST_DIR, "L2SVM",
                new String[] { "w" }));
	}
	
	@Test
    public void testL2SVM(){
		int rows = 1000;
        int cols = 100;
        double epsilon = 1.0e-8;
        double lambda = 1.0;
        int maxiterations = 3;
        int maxNumberOfMRJobs = 21;
        
        /* this is old way of running the test by passing the variable over to the dml script */
        TestConfiguration config = getTestConfiguration(TEST_L2SVM);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("eps", Math.pow(10, -8));
        config.addVariable("lambda", lambda);
      
        /* This is for running the junit test by constructing the arguments directly */
		String L2SVM_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = L2SVM_HOME + TEST_L2SVM + ".dml";
		programArgs = new String[]{"-args", 
	               				L2SVM_HOME + INPUT_DIR + "X", 
	               				L2SVM_HOME + INPUT_DIR + "Y", 
	               				Integer.toString(rows), 
	               				Integer.toString(cols),
	               				Double.toString(epsilon), 
	               				Double.toString(lambda), 
	               				Integer.toString(maxiterations),
	               				L2SVM_HOME + OUTPUT_DIR + "w"};
		/*dmlArgs = new String[]{"-f", 
							   L2SVM_HOME + TEST_L2SVM + ".dml",
				               "-args", 
				               L2SVM_HOME + INPUT_DIR + "X", 
				               L2SVM_HOME + INPUT_DIR + "Y", 
				               Integer.toString(rows), 
				               Integer.toString(cols),
				               Double.toString(epsilon), 
				               Double.toString(lambda), 
				               Integer.toString(maxiterations),
				               L2SVM_HOME + OUTPUT_DIR + "w"};
	
		dmlArgsDebug = new String[]{"-f", 
									L2SVM_HOME + TEST_L2SVM + ".dml", 
									"-args", 
									L2SVM_HOME + INPUT_DIR + "X", 
                                    L2SVM_HOME + INPUT_DIR + "Y", 
                                    Integer.toString(rows), 
                                    Integer.toString(cols),
                                    Double.toString(Math.pow(10, -8)), 
                                    Double.toString(lambda), 
                                    Integer.toString(maxiterations), 
                                    L2SVM_HOME + OUTPUT_DIR + "w"};*/
		
		fullRScriptName = L2SVM_HOME + TEST_L2SVM + ".R";
		rCmd = "Rscript" + " " + 
				fullRScriptName + " " + 
				L2SVM_HOME + INPUT_DIR + " " + 
				Double.toString(epsilon) + " " + 
				Double.toString(lambda) + " " + 
				Integer.toString(maxiterations) + " " + 
				L2SVM_HOME + EXPECTED_DIR;
		
        loadTestConfiguration(config);

        double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.01, -1);
        double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, -1);
        for(int i=0; i<rows; i++)
        	Y[i][0] = (Y[i][0] > 0) ? 1 : -1;
        
        writeInputMatrix("X", X, true);
        writeInputMatrix("Y", Y, true);
     
        boolean exceptionExpected = false;
		
        runTest(true, exceptionExpected, null, maxNumberOfMRJobs);
		
		runRScript(true);
		disableOutAndExpectedDeletion();
        
		HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        boolean success = TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
        //System.out.println(success+"");
	}
}
