package dml.test.integration.applications;

import org.junit.Test;

import java.util.HashMap;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class L2SVMTest extends AutomatedTestBase {
	private final static String TEST_DIR = "applications/l2svm/";
	private final static String TEST_L2SVM = "L2SVMTest";

	@Override
	public void setUp() {
		setUpBase();
    	addTestConfiguration(TEST_L2SVM, new TestConfiguration(TEST_DIR, "L2SVMTest",
                new String[] { "w" }));
	}
	
	@Test
    public void testL2SVM(){
		int rows = 1000;
        int cols = 100;

        TestConfiguration config = getTestConfiguration(TEST_L2SVM);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("eps", Math.pow(10, -8));
        config.addVariable("lambda", 1);
      
        loadTestConfiguration(config);

        double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.01, -1);
        double[][] y = getRandomMatrix(rows, 1, -1, 1, 1, -1);
        for(int i=0; i<rows; i++)
        	y[i][0] = (y[i][0] > 0) ? 1 : -1;
        
        writeInputMatrix("X", X, true);
        writeInputMatrix("y", y, true);
        
        runTest();
        runRScript();
        
		HashMap<CellIndex, Double> wR = readRMatrixFromFS("w");
        HashMap<CellIndex, Double> wDML= readDMLMatrixFromHDFS("w");
        boolean success = TestUtils.compareMatrices(wR, wDML, Math.pow(10, -14), "wR", "wDML");
        //System.out.println(success+"");
	}
}
