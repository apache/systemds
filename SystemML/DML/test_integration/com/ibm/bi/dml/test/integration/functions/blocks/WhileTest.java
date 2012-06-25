package com.ibm.bi.dml.test.integration.functions.blocks;

import static junit.framework.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>inner loop computation</li>
 * 	<li>clean up</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class WhileTest extends AutomatedTestBase {

    @Override
    public void setUp() {
        baseDirectory = SCRIPT_DIR + "functions/blocks/";
        
        // positive tests
        availableTestConfigurations.put("ComputationTest", new TestConfiguration("WhileTest",
        		new String[] { "b" }));
        availableTestConfigurations.put("CleanUpTest", new TestConfiguration("WhileTest",
        		new String[] { "b" }));
        
        // negative tests
    }
    
    @Test
    public void testComputation() {
    	int rows = 10;
    	int cols = 10;
        int maxIterations = 3;
        
    	TestConfiguration config = availableTestConfigurations.get("ComputationTest");
    	config.addVariable("rows", rows);
    	config.addVariable("cols", cols);
    	config.addVariable("maxiterations", maxIterations);
    	
    	loadTestConfiguration("ComputationTest");
        
        double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
        double[][] b = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                b[i][j] = Math.pow(3, maxIterations) * a[i][j];
            }
        }
        
        writeInputMatrix("a", a);
        writeExpectedMatrix("b", b);
        
        runTest();
        
        compareResults(1e-14);
    }
    
    @Test
    public void testCleanUp() {
    	int rows = 10;
    	int cols = 10;
        int maxIterations = 3;
        
    	TestConfiguration config = availableTestConfigurations.get("CleanUpTest");
    	config.addVariable("rows", rows);
    	config.addVariable("cols", cols);
    	config.addVariable("maxiterations", maxIterations);
    	
    	loadTestConfiguration("CleanUpTest");
        
        TestUtils.removeTemporaryFiles();
        
        createRandomMatrix("a", rows, cols, -1, 1, 1, -1);
        
        runTest();
        
        assertFalse("not all temp directories were removed", TestUtils.checkForTemporaryFiles());
    }
}
