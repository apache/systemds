/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;




public class CastAsScalarTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	    
    private final static String TEST_DIR = "functions/unary/matrix/";
    private final static String TEST_GENERAL = "General";
    

    @Override
    public void setUp() {
        addTestConfiguration(TEST_GENERAL, new TestConfiguration(TEST_DIR, "CastAsScalarTest", new String[] { "b" }));
    }
    
    @Test
    public void testGeneral() {
        loadTestConfiguration(TEST_GENERAL);
        
        createHelperMatrix();
        writeInputMatrix("a", new double[][] { { 2 } });
        writeExpectedHelperMatrix("b", 2);
        
        runTest();
        
        compareResults();
    }

}
