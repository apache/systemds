package dml.test.integration.scalability;

import org.junit.Test;

import dml.test.integration.AutomatedScalabilityTestBase;
import dml.test.integration.TestConfiguration;


public class PageRankTest extends AutomatedScalabilityTestBase {

    @Override
    public void setUp() {
        baseDirectory = "test/scripts/scalability/page_rank/";
        availableTestConfigurations.put("PageRankTest", new TestConfiguration("PageRankTest", new String[] { "p" }));
        matrixSizes = new int[][] {
                { 9914 }
        };
    }
    
    @Test
    public void testPageRank() {
    	loadTestConfiguration("PageRankTest");
        
        addInputMatrix("g", -1, -1, 1, 1, 0.000374962, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(0);
        addInputMatrix("p", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("e", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("u", 1, -1, 1, 1, 1, -1).setColsIndexInMatrixSizes(0);
        
        runTest();
    }

}
