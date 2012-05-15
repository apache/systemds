package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;




public class CastAsScalarTest extends AutomatedTestBase {
    
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
