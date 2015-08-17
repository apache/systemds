/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.blocks;

import org.junit.Assert;
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
public class WhileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final String TEST_DIR = "functions/blocks/";

    @Override
    public void setUp() {
        // positive tests
        addTestConfiguration("ComputationTest", new TestConfiguration(TEST_DIR, "WhileTest",
        		new String[] { "b" }));
        addTestConfiguration("CleanUpTest", new TestConfiguration(TEST_DIR, "WhileTest",
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
    	
    	loadTestConfiguration(config);
        
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
    	
    	loadTestConfiguration(config);
        
        TestUtils.removeTemporaryFiles();
        
        createRandomMatrix("a", rows, cols, -1, 1, 1, -1);
        
        runTest();
        
        Assert.assertFalse("not all temp directories were removed", TestUtils.checkForTemporaryFiles());
    }
}
