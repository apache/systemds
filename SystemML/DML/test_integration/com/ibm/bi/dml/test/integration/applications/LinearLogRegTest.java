package com.ibm.bi.dml.test.integration.applications;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


@RunWith(value = Parameterized.class)
public class LinearLogRegTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "applications/linearLogReg/";
    private final static String TEST_LINEAR_LOG_REG = "LinearLogReg";

    private int numRecords, numFeatures, numTestRecords;
    
	public LinearLogRegTest(int numRecords, int numFeatures, int numTestRecords) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
		this.numTestRecords = numTestRecords;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { {100, 50, 25}, {1000, 500, 200}, {10000, 750, 1500}};
	   return Arrays.asList(data);
	 }

    @Override
    public void setUp()
    {
    	setUpBase();
    	addTestConfiguration(TEST_LINEAR_LOG_REG, new TestConfiguration(TEST_DIR, TEST_LINEAR_LOG_REG,
                new String[] { "w" }));
    }
    
    @Test
    public void testLinearLogReg() throws ClassNotFoundException, SQLException, IOException
    {
    	int rows = numRecords;			// # of rows in the training data 
        int cols = numFeatures;
        int rows_test = numTestRecords; // # of rows in the test data 
        int cols_test = cols; 			// # of rows in the test data 

        TestConfiguration config = getTestConfiguration(TEST_LINEAR_LOG_REG);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("rows_test", rows_test);
        config.addVariable("cols_test", cols_test);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String LLR_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = LLR_HOME + TEST_LINEAR_LOG_REG + ".dml";
		programArgs = new String[]{"-args", LLR_HOME + INPUT_DIR + "X" , 
				                        Integer.toString(rows), Integer.toString(cols),
				                         LLR_HOME + INPUT_DIR + "Xt" , 
				                        Integer.toString(rows_test), Integer.toString(cols_test),
				                         LLR_HOME + INPUT_DIR + "y" ,
				                         LLR_HOME + INPUT_DIR + "yt" ,
				                         LLR_HOME + OUTPUT_DIR + "w" };
		
		fullRScriptName = LLR_HOME + TEST_LINEAR_LOG_REG + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       LLR_HOME + INPUT_DIR + " " + LLR_HOME + EXPECTED_DIR;
      
        loadTestConfiguration(TEST_LINEAR_LOG_REG);

        // prepare training data set
        double[][] X = getRandomMatrix(rows, cols, 1, 10, 1, 100);
        double[][] y = getRandomMatrix(rows, 1, 0.01, 1, 1, 100);
        writeInputMatrix("X", X, true);
        writeInputMatrix("y", y, true);
        
        // prepare test data set
        double[][] Xt = getRandomMatrix(rows_test, cols_test, 1, 10, 1, 100);
        double[][] yt = getRandomMatrix(rows_test, 1, 0.01, 1, 1, 100);
        writeInputMatrix("Xt", Xt, true);
        writeInputMatrix("yt", yt, true);
        
        
        HashMap<CellIndex, Double> wSQL = null;
        if(RUNNETEZZA)
        {
        	String path1 = baseDirectory + INPUT_DIR + "X";
			String path2 = baseDirectory + INPUT_DIR + "y";
			String path3 = baseDirectory + INPUT_DIR + "Xt";
			String path4 = baseDirectory + INPUT_DIR + "yt";
			String file1 = DMLScript.class.getProtectionDomain().getCodeSource().getLocation().toString().replace("bin/", "") + path1.replace("./", "");
			String file2 = DMLScript.class.getProtectionDomain().getCodeSource().getLocation().toString().replace("bin/", "") + path2.replace("./", "");
			String file3 = DMLScript.class.getProtectionDomain().getCodeSource().getLocation().toString().replace("bin/", "") + path3.replace("./", "");
			String file4 = DMLScript.class.getProtectionDomain().getCodeSource().getLocation().toString().replace("bin/", "") + path4.replace("./", "");
			
			NetezzaConnector con = new NetezzaConnector();
			con.connect();
			
			con.exportHadoopDirectoryToNetezza(file1.substring(6).replace("%20", " "), path1, true);
			con.exportHadoopDirectoryToNetezza(file2.substring(6).replace("%20", " "), path2, true);
			con.exportHadoopDirectoryToNetezza(file3.substring(6).replace("%20", " "), path3, true);
			con.exportHadoopDirectoryToNetezza(file4.substring(6).replace("%20", " "), path4, true);
			con.disconnect();
			
			runSQL();
			wSQL = readDMLmatrixFromTable("w");
        }
        
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Rand - 1 job 
		 * Computation before while loop - 4 jobs
		 * While loop iteration - 9 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 31;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
        
		runRScript(true);
        
        HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
        
        if(wSQL != null)
        {
        	TestUtils.compareMatrices(wR, wSQL, Math.pow(10, -14), "wR", "wSQL");
        }
    }
}
