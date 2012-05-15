package com.ibm.bi.dml.test.integration.scalability;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedScalabilityTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class LinearRegressionTest extends AutomatedScalabilityTestBase {

    @Override
    public void setUp() {
        baseDirectory = "test/scripts/scalability/linear_regression/";
        availableTestConfigurations.put("LinearRegressionTest", new TestConfiguration("LinearRegressionTest",
        		new String[] { "w" }));
        matrixSizes = new int[][] {
                { 19004, 15436 }
        };
    }
    
    @Test
    public void testLinearRegression() {
    	loadTestConfiguration("LinearRegressionTest");
        
        addInputMatrix("g", -1, -1, 0, 1, 0.00594116, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(1);
        addInputMatrix("b", -1, 1, 1, 10, 1, -1).setRowsIndexInMatrixSizes(0);
        
        runTest();
    }

}
