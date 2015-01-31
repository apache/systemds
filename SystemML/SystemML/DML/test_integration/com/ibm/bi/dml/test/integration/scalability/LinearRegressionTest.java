/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.scalability;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedScalabilityTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class LinearRegressionTest extends AutomatedScalabilityTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "test/scripts/scalability/linear_regression/";
	
    @Override
    public void setUp() {
    	TestUtils.clearAssertionInformation();
        addTestConfiguration("LinearRegressionTest", new TestConfiguration(TEST_DIR, "LinearRegressionTest", new String[] { "w" }));
        matrixSizes = new int[][] {
                { 19004, 15436 }
        };
    }
    
    @Test
    public void testLinearRegression() {
    	TestConfiguration config = getTestConfiguration("LinearRegressionTest");
    	loadTestConfiguration(config);
        
        addInputMatrix("g", -1, -1, 0, 1, 0.00594116, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(1);
        addInputMatrix("b", -1, 1, 1, 10, 1, -1).setRowsIndexInMatrixSizes(0);
        
        runTest();
    }

}
