package dml.test.integration.applications;

import java.io.IOException;
import java.sql.SQLException;
import java.util.HashMap;

import org.junit.Test;

import dml.api.DMLScript;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.sql.sqlcontrolprogram.NetezzaConnector;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class LinearLogRegTest extends AutomatedTestBase
{

    private final static String TEST_DIR = "applications/linearLogReg/";
    private final static String TEST_LINEAR_LOG_REG = "LinearLogRegTest";


    @Override
    public void setUp()
    {
    	setUpBase();
    	addTestConfiguration(TEST_LINEAR_LOG_REG, new TestConfiguration(TEST_DIR, "LinearLogRegTest",
                new String[] { "w" }));
    }
    
    @Test
    public void testLinearLogReg() throws ClassNotFoundException, SQLException, IOException
    {
    	int rows = 100;	// // # of rows in the training data 
        int cols = 50;
        int rows_test = 25; 	// # of rows in the test data 
        int cols_test = cols; 	// # of rows in the test data 

        TestConfiguration config = getTestConfiguration(TEST_LINEAR_LOG_REG);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("rows_test", rows_test);
        config.addVariable("cols_test", cols_test);
      
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
		//int expectedNumberOfJobs = 31;
		runTest();
        
		runRScript();
        
        HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
        
        if(wSQL != null)
        {
        	TestUtils.compareMatrices(wR, wSQL, Math.pow(10, -14), "wR", "wSQL");
        }
    }
}
